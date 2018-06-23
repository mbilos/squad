import tensorflow as tf
import math

def get_shape(x):
    static = x.get_shape().as_list()
    shape = tf.shape(x)
    return [static[i] or shape[i] for i in range(len(static))]

def layer_norm(inputs, scope='layer-norm', epsilon=1e-8, reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        shape = inputs.get_shape().as_list()

        scale = tf.get_variable('scale', shape[-1], initializer=tf.ones_initializer())
        bias = tf.get_variable('bias', shape[-1], initializer=tf.zeros_initializer())

        mean, variance = tf.nn.moments(inputs, -1, keep_dims=True)
        normalized = (inputs - mean) / tf.sqrt(variance + epsilon)

        return normalized * scale + bias

def birnn(inputs, length, dim, cell_type='gru', dropout=0.0, scope='bi-rnn', reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        outputs, states = tf.nn.bidirectional_dynamic_rnn(
            tf.nn.rnn_cell.DropoutWrapper(tf.contrib.rnn.GRUCell(dim), input_keep_prob=1.0 - dropout),
            tf.nn.rnn_cell.DropoutWrapper(tf.contrib.rnn.GRUCell(dim), input_keep_prob=1.0 - dropout),
            inputs,
            sequence_length=length,
            dtype=tf.float32)

        outputs = tf.concat(outputs, -1)
        return outputs

def highway(x, dim, dropout=0.0, scope='highway', reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        with tf.variable_scope('out', reuse=reuse):
            out = tf.layers.dense(x, dim, tf.nn.relu, reuse=reuse)
            out = tf.nn.dropout(out, 1.0 - dropout)
        with tf.variable_scope('out', reuse=reuse):
            keep = tf.layers.dense(x, dim, tf.nn.sigmoid, bias_initializer=tf.constant_initializer(-2), reuse=reuse)

        return (1 - keep) * x + keep * out

def gated_connection(prev, current, scope='gated-connection', reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        shape = current.get_shape().as_list()

        keep = tf.layers.dense(
            current,
            shape[-1],
            activation=tf.sigmoid,
            bias_initializer=tf.constant_initializer(-2),
            reuse=reuse)

        return (1 - keep) * prev + keep * current

def trilinear(c, q, scope='trilinear', reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        shape = get_shape(c)

        w_dot = tf.get_variable('w-dot', [1, 1, shape[-1]], tf.float32)
        c_dot = c * w_dot

        with tf.variable_scope('c-proj', reuse=reuse):
            c_proj = tf.layers.dense(c, 1, use_bias=False, reuse=reuse)
        with tf.variable_scope('q-proj', reuse=reuse):
            q_proj = tf.transpose(tf.layers.dense(q, 1, use_bias=False, reuse=reuse), [0, 2, 1])

        c_q = tf.matmul(c_dot, q, transpose_b=True)

        return c_proj + q_proj + c_q

def bi_attention(c, q, similarity, c_mask=None, q_mask=None, only_c2q=False, scope='bidaf', reuse=None):

    def mask(similarity, c_mask, q_mask, mode='add', big_number=1e30):
        shape = get_shape(similarity)

        c_mask = tf.tile(tf.expand_dims(c_mask, 2), [1, 1, shape[2]])
        q_mask = tf.tile(tf.expand_dims(q_mask, 1), [1, shape[1], 1])
        mask = tf.cast(c_mask * q_mask, tf.float32)

        if mode == 'add':
            mask = 1 - tf.sign(mask)
            similarity -= big_number * mask
        else:
            similarity *= mask

        return similarity

    with tf.variable_scope(scope, reuse=reuse):
        add_mask = lambda x: mask(x, c_mask, q_mask)
        mul_mask = lambda x: mask(x, c_mask, q_mask, 'mul')

        row_norm = mul_mask(tf.nn.softmax(add_mask(similarity), -1))
        A = tf.matmul(row_norm, q) # context to query

        if only_c2q:
            return A

        column_norm = mul_mask(tf.nn.softmax(add_mask(similarity), 1))
        B = tf.matmul(tf.matmul(row_norm, column_norm, transpose_b=True), c) # query to context

        attention = tf.concat([c, A, c * A, c * B], -1)
        return attention

def char_embed(inputs, embed, dropout=0.0, scope='char-embedding', reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        vocab, dim = get_shape(embed)

        c = tf.nn.embedding_lookup(embed, inputs)
        c = tf.nn.dropout(c, keep_prob=1.0 - dropout)
        c = tf.layers.conv2d(c, dim, kernel_size=[1, 5], activation=tf.nn.relu)
        c = tf.reduce_max(c, axis=2)

        return c

def sfu(inputs, fusion, scope='semantic-fusion-unit', reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        shape = get_shape(inputs)
        x = tf.concat([inputs] + fusion, -1)

        with tf.variable_scope('res', reuse=reuse):
            res = tf.layers.dense(x, shape[-1], activation=tf.nn.tanh, reuse=reuse)
        with tf.variable_scope('gate', reuse=reuse):
            gate = tf.layers.dense(x, shape[-1], activation=tf.nn.sigmoid, reuse=reuse)

        return gate * res + (1 - gate) * inputs

def positional_encoding(inputs, min_timescale=1.0, max_timescale=1.0e4, start_index=0):
    # from https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/layers/common_attention.py
    _, length, channels = get_shape(inputs)

    position = tf.to_float(tf.range(length) + start_index)
    num_timescales = channels // 2
    log_timescale_increment = (math.log(float(max_timescale) / float(min_timescale)) / (tf.to_float(num_timescales) - 1))
    inv_timescales = min_timescale * tf.exp(tf.to_float(tf.range(num_timescales)) * -log_timescale_increment)
    scaled_time = tf.expand_dims(position, 1) * tf.expand_dims(inv_timescales, 0)
    signal = tf.concat([tf.sin(scaled_time), tf.cos(scaled_time)], axis=1)
    signal = tf.pad(signal, [[0, 0], [0, tf.mod(channels, 2)]])
    signal = tf.reshape(signal, [1, length, channels])

    return inputs + signal

def encoder_block(inputs, num_blocks, num_convolutions, kernel, mask, dropout=0.0, scope='encoder', reuse=None):

    def layer_dropout(prev, residual, dropout):
        pred = tf.random_uniform([], 0.0, 1.0) < dropout
        return tf.cond(pred, lambda: prev, lambda: prev + residual)

    with tf.variable_scope(scope, reuse=reuse):
        block = [inputs]

        for i in range(num_blocks):
            with tf.variable_scope('encoder-block-%d' % i, reuse=reuse):
                dim = get_shape(block[i])[-1]

                pos = positional_encoding(block[i])

                conv = [pos]
                for j in range(num_convolutions):
                    with tf.variable_scope('residual-block-%d' % j, reuse=reuse):
                        x = layer_norm(conv[j], reuse=reuse)
                        if (j + 1) % 2 == 0:
                            x = tf.nn.dropout(x, 1.0 - dropout)
                        x = tf.layers.conv1d(x, dim, kernel, padding='same', activation=tf.nn.relu, reuse=reuse)
                        res = layer_dropout(conv[j], x, (j + 1) / num_convolutions * dropout)
                        conv.append(res)

                with tf.variable_scope('self-attention', reuse=reuse) as scope:
                    x = layer_norm(conv[-1], reuse=reuse)

                    similarity = trilinear(x, x, reuse=reuse)
                    attention = bi_attention(x, x, similarity, mask, mask, only_c2q=True, reuse=reuse)
                    res = tf.layers.dense(res, dim, activation=tf.nn.relu)

                    self_attention = layer_dropout(conv[-1], res, (i + 1) / num_blocks * dropout)

                with tf.variable_scope('feedforward', reuse=reuse):
                    x = layer_norm(self_attention, reuse=reuse)
                    x = tf.layers.dense(x, dim, activation=tf.nn.relu, reuse=reuse)
                    res = tf.nn.dropout(x, 1.0 - dropout)
                    ff = layer_dropout(self_attention, res, (i + 1) / num_blocks * dropout)

                block.append(ff)

        return block[-1]
