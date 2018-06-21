import tensorflow as tf
from tensorflow.contrib.rnn import BasicLSTMCell, GRUCell, DropoutWrapper

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
        cell = GRUCell if cell_type == 'gru' else BasicLSTMCell

        fw = DropoutWrapper(cell(dim), input_keep_prob=1.0 - dropout)
        bw = DropoutWrapper(cell(dim), input_keep_prob=1.0 - dropout)

        outputs, states = tf.nn.bidirectional_dynamic_rnn(fw, bw, inputs, length, dtype=tf.float32)
        outputs = tf.concat(outputs, -1)
        return outputs

def highway(x, dropout=0.0, scope='highway', reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        shape = get_shape(x)

        out = tf.layers.dense(x, shape[-1], tf.nn.relu, reuse=reuse)
        out = tf.layers.dropout(out, rate=dropout)
        keep = tf.layers.dense(x, shape[-1], tf.nn.sigmoid, bias_initializer=tf.constant_initializer(-1), reuse=reuse)

        return (1 - keep) * x + keep * out

def trilinear(c, q, scope='trilinear', reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        shape = get_shape(c)

        w_dot = tf.get_variable('w-dot', [1, 1, shape[-1]], tf.float32)
        c_dot = c * w_dot

        c_proj = tf.layers.dense(c, 1, use_bias=False)
        q_proj = tf.transpose(tf.layers.dense(q, 1, use_bias=False), [0, 2, 1])

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

        res = tf.layers.dense(x, shape[-1], activation=tf.nn.tanh)
        gate = tf.layers.dense(x, shape[-1], activation=tf.nn.sigmoid)

        return gate * res + (1 - gate) * inputs