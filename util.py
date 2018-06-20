import numpy as np
import tensorflow as tf
import math

from tensorflow.python.ops.rnn_cell import BasicLSTMCell, GRUCell

def get_shape(x):
    static = x.get_shape().as_list()
    shape = tf.shape(x)
    return [static[i] or shape[i] for i in range(len(static))]


def bidirectional_dynamic_rnn(inputs, sequence_length, hidden_size, dropout=0.0, cell_type='gru', concat=False, scope='bi-rnn', reuse=None):

    if cell_type == 'gru':
        cell = GRUCell(hidden_size)
    elif cell_type == 'lstm':
        cell = BasicLSTMCell(hidden_size)
    else:
        raise NotImplementedError('Invalid cell name')

    with tf.variable_scope(scope, reuse=reuse):
        outputs, state = tf.nn.bidirectional_dynamic_rnn(
            tf.nn.rnn_cell.DropoutWrapper(cell, input_keep_prob=1.0 - dropout), # forward
            tf.nn.rnn_cell.DropoutWrapper(cell, input_keep_prob=1.0 - dropout), # backward
            inputs,
            dtype=tf.float32,
            sequence_length=sequence_length)

        if concat:
            outputs = tf.concat(outputs, -1)

        return outputs, state

def multihead_attention(Q, K, V, heads=1, mask=None, dropout=0.0, scope='multihead-attention', reuse=None, training=None):
    '''
    Applies multihead attention as described in https://arxiv.org/abs/1706.03762
    And as implemented in https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/layers/common_attention.py

    Args:
        Q: query tensor (batch, len_q, dim_k)
        K: key tensor   (batch, len_k, dim_k)
        V: value tensor (batch, len_k, dim_v)
        heads: number of heads (must be divisor of dim_k and dim_v) (default: 1)
        mask: tensor (batch, max_len) of 0 and 1 masking input sequences (default: None)
        dropout: percentage of units to randomly drop
        scope: name ot the scope (default: 'multihead-attention')
        reuse: whether to reuse the weights (default: None)
    Returns:
        tensor with same shape as V
    '''
    with tf.variable_scope(scope, reuse=reuse):
        k_shape = get_shape(K)
        v_shape = get_shape(V)

        assert k_shape[-1] % heads == 0 and v_shape[-1] % heads == 0
        assert k_shape[1] == v_shape[1]

        def linear(inputs, shape, scope):
            with tf.variable_scope(scope, reuse=reuse):
                inputs = tf.layers.conv1d(inputs, shape[-1], 1, use_bias=False, padding='SAME')
                inputs = tf.layers.dropout(inputs, rate=dropout, training=training)
                inputs = tf.reshape(inputs, shape[:-1] + [heads, shape[-1] // heads])
                return tf.transpose(inputs, [0, 2, 1, 3]) # [batch, heads, max_sentence_len, embedding / heads]

        Q, K, V = linear(Q, k_shape, 'query'), linear(K, k_shape, 'key'), linear(V, v_shape, 'value')
        Q *= tf.rsqrt(tf.to_float(k_shape[-1] // heads))

        # [batch, heads, max_sentence_len, embedding / heads]
        alpha = tf.matmul(Q, K, transpose_b=True)

        if mask is not None:
            # [batch, max_len] -> [batch, heads, max_len, max_len]
            mask = tf.tile(tf.reshape(mask, [-1, 1, 1, k_shape[1]]), [1, heads, k_shape[1], 1])
            # 1 -> 0 | 0 -> -1e30 | softmax(-1e30) -> 0
            mask = (1 - tf.sign(tf.abs(tf.cast(mask, tf.float32)))) * (-1e30)
            alpha += mask

        alpha = tf.nn.softmax(alpha)
        attended = tf.matmul(alpha, V)

        # [batch, max_sentence_len, embedding]
        attended = tf.transpose(attended, [0, 2, 1, 3])
        a_shape = get_shape(attended)
        attended = tf.reshape(attended, a_shape[:-2] + [a_shape[-2] * a_shape[-1]])

        attended = tf.layers.conv1d(attended, v_shape[-1], 1, use_bias=False, padding='SAME')
        attended = tf.layers.dropout(attended, rate=dropout, training=training)

        return attended

def _multihead_attention(queries, units, num_heads,
                        memory = None,
                        scope = "Multi_Head_Attention",
                        reuse = None,
                        mask = None,
                        is_training = True,
                        bias = True,
                        dropout = 0.0):
    # copied from https://github.com/NLPLearn/QANet/blob/master/layers.py
    def split_last_dimension(x, n):
        old_shape = x.get_shape().dims
        last = old_shape[-1]
        new_shape = old_shape[:-1] + [n] + [last // n if last else None]
        ret = tf.reshape(x, tf.concat([tf.shape(x)[:-1], [n, -1]], 0))
        ret.set_shape(new_shape)
        return tf.transpose(ret,[0,2,1,3])

    def combine_last_two_dimensions(x):
        old_shape = x.get_shape().dims
        a, b = old_shape[-2:]
        new_shape = old_shape[:-2] + [a * b if a and b else None]
        ret = tf.reshape(x, tf.concat([tf.shape(x)[:-2], [-1]], 0))
        ret.set_shape(new_shape)
        return ret

    def dot_product_attention(q,
                          k,
                          v,
                          bias,
                          mask = None,
                          is_training = True,
                          scope=None,
                          reuse = None,
                          dropout = 0.0):
        with tf.variable_scope(scope, default_name="dot_product_attention", reuse = reuse):
            logits = tf.matmul(q, k, transpose_b=True)
            if bias:
                b = tf.get_variable("bias",
                        logits.shape[-1],
                        regularizer=regularizer,
                        initializer = tf.zeros_initializer())
                logits += b
            if mask is not None:
                shapes = [x  if x != None else -1 for x in logits.shape.as_list()]
                mask = tf.reshape(mask, [shapes[0],1,1,shapes[-1]])
                logits = mask_logits(logits, mask)
            weights = tf.nn.softmax(logits, name="attention_weights")
            weights = tf.nn.dropout(weights, 1.0 - dropout)
            return tf.matmul(weights, v)

    with tf.variable_scope(scope, reuse = reuse):
        if memory is None:
            memory = queries

        memory = tf.layers.conv1d(memory, 2 * units, 1, name="memory_projection", reuse = reuse)
        query = tf.layers.conv1d(queries, units, 1, name="query_projection", reuse = reuse)
        Q = split_last_dimension(query, num_heads)
        K, V = [split_last_dimension(tensor, num_heads) for tensor in tf.split(memory,2,axis = 2)]

        key_depth_per_head = units // num_heads
        Q *= key_depth_per_head**-0.5
        x = dot_product_attention(Q,K,V,
                                  bias = bias,
                                  mask = mask,
                                  is_training = is_training,
                                  scope = "dot_product_attention",
                                  reuse = reuse, dropout = dropout)
        return combine_last_two_dimensions(tf.transpose(x,[0,2,1,3]))


def trilinear(c, q, scope='trilinear', reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        shape = get_shape(c)

        w_dot = tf.get_variable('w-dot', [1, 1, shape[-1]], tf.float32)
        c_dot = c * w_dot

        c_proj = tf.layers.dense(c, 1, use_bias=False)
        q_proj = tf.transpose(tf.layers.dense(q, 1, use_bias=False), [0, 2, 1])

        c_q = tf.matmul(c_dot, q, transpose_b=True)

        return c_proj + q_proj + c_q


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


def norm(inputs, epsilon=1e-8, reuse=None):
    mean, variance = tf.nn.moments(inputs, -1, keep_dims=True)
    return (inputs - mean) / tf.sqrt(variance + epsilon)


def layer_norm(inputs, scope='layer-norm', epsilon=1e-8, reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        shape = inputs.get_shape().as_list()

        scale = tf.get_variable('scale', shape[-1], initializer=tf.ones_initializer())
        bias = tf.get_variable('bias', shape[-1], initializer=tf.zeros_initializer())

        mean, variance = tf.nn.moments(inputs, -1, keep_dims=True)
        normalized = (inputs - mean) / tf.sqrt(variance + epsilon)

        return normalized * scale + bias


def depthwise_separable_convolution_old(inputs, filters, kernel_size, dropout=0.0, scope='conv', reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        inputs = tf.expand_dims(inputs, 2)
        conv = tf.layers.separable_conv2d(inputs,
                                          filters,
                                          [kernel_size, 1],
                                          padding='same',
                                          activation=tf.nn.relu,
                                          reuse=reuse,
                                          use_bias=False)
        conv = tf.squeeze(conv, 2)
        return tf.nn.dropout(conv, 1.0 - dropout)


def depthwise_separable_conv(inputs, filters, kernel_size, activation=None, dropout=0.0, scope='conv', reuse=None, training=None):
    with tf.variable_scope(scope, reuse=reuse):
        shape = get_shape(inputs)

        with tf.variable_scope('depthwise', reuse=reuse):
            depthwise = tf.layers.conv1d(inputs, shape[-1], kernel_size, padding='same', reuse=reuse)
        with tf.variable_scope('pointwise', reuse=reuse):
            pointwise = tf.layers.conv1d(depthwise, filters, 1, activation=activation, padding='same', reuse=reuse)

        return tf.layers.dropout(pointwise, rate=dropout, training=training)


def mask_logits(inputs, mask, mask_value=-1e30):
    mask = tf.cast(mask, tf.float32)
    return inputs + mask_value * (1 - mask)


def positional_encoding_old(inputs, dropout=0.0):
    # from https://github.com/Kyubyong/transformer/blob/master/modules.py#L141
    shape = inputs.get_shape().as_list()

    position = np.array([
        [pos / np.power(10000, 2.*i/shape[-1]) for i in range(shape[-1])]
        for pos in range(shape[1])])

    position[:, 0::2] = np.sin(position[:, 0::2]) # dim 2i
    position[:, 1::2] = np.cos(position[:, 1::2]) # dim 2i+1

    position = tf.convert_to_tensor(position, dtype=tf.float32)

    return tf.nn.dropout(inputs + position, 1 - dropout)

def positional_encoding(inputs, dropout=0.0):
    # from https://github.com/hengruo/QANet-pytorch/blob/master/models.py#L38
    _, d, emb = inputs.get_shape().as_list()

    pos = tf.tile(tf.expand_dims(tf.range(emb, dtype=tf.float32), 0), [d, 1])

    phase = [0 if i % 2 == 0 else math.pi / 2 for i in range(d)]
    phase = tf.expand_dims(tf.convert_to_tensor(phase), -1)

    freq = [10000 ** (-i / d) if i % 2 == 0 else -10000 ** (-(i - 1) / d) for i in range(d)]
    freq = tf.expand_dims(tf.convert_to_tensor(freq), -1)

    position_encoder = tf.sin(pos * freq + phase)

    return tf.sin(position_encoder) + inputs

