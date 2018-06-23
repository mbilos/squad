import tensorflow as tf
import layers

from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.framework import ops
from tensorflow.python.training import optimizer

class AdamaxOptimizer(optimizer.Optimizer):
    # from https://github.com/openai/iaf/blob/master/tf_utils/adamax.py
    def __init__(self, learning_rate=2e-3, beta1=0.9, beta2=0.999, use_locking=False, name="Adamax"):
        super(AdamaxOptimizer, self).__init__(use_locking, name)
        self._lr = learning_rate
        self._beta1 = beta1
        self._beta2 = beta2

        self._lr_t = None
        self._beta1_t = None
        self._beta2_t = None

    def _prepare(self):
        self._lr_t = ops.convert_to_tensor(self._lr, name="learning_rate")
        self._beta1_t = ops.convert_to_tensor(self._beta1, name="beta1")
        self._beta2_t = ops.convert_to_tensor(self._beta2, name="beta2")

    def _create_slots(self, var_list):
        for v in var_list:
            self._zeros_slot(v, "m", self._name)
            self._zeros_slot(v, "v", self._name)

    def _apply_dense(self, grad, var):
        lr_t = math_ops.cast(self._lr_t, var.dtype.base_dtype)
        beta1_t = math_ops.cast(self._beta1_t, var.dtype.base_dtype)
        beta2_t = math_ops.cast(self._beta2_t, var.dtype.base_dtype)
        eps = 1e-8

        v = self.get_slot(var, "v")
        v_t = v.assign(beta1_t * v + (1. - beta1_t) * grad)
        m = self.get_slot(var, "m")
        m_t = m.assign(tf.maximum(beta2_t * m + eps, tf.abs(grad)))
        g_t = v_t / m_t

        var_update = state_ops.assign_sub(var, lr_t * g_t)
        return control_flow_ops.group(*[var_update, m_t, v_t])

    def _apply_sparse(self, grad, var):
        raise NotImplementedError("Sparse gradient updates are not supported.")

class MnemonicReader:
    def __init__(self, config):
        self.config = config

        self.dropout = tf.get_variable('dropout', shape=[], dtype=tf.float32, initializer=tf.constant_initializer(self.config.dropout), trainable=False)
        self.global_step = tf.get_variable('global_step', shape=[], dtype=tf.int32, initializer=tf.constant_initializer(0), trainable=False)

        self.lr = tf.get_variable('learning-rate', shape=[], dtype=tf.float32, initializer=tf.constant_initializer(self.config.learning_rate), trainable=False)
        self.decay_lr = tf.assign(self.lr, tf.maximum(self.lr / 2, 1e-6))

        self.word_embed = tf.get_variable("word-emb-matrix", shape=self.config.embed.shape, initializer=tf.constant_initializer(self.config.embed), trainable=False)
        self.char_embed = tf.get_variable('char-emb-matrix', [self.config.unique_chars, self.config.char_embed])

        self.aligner_hops = 2
        self.pointer_hops = 2

        self.forward()

    def forward(self):
        self.c_words = tf.placeholder(tf.int32, [None, self.config.context_len], 'context-words')
        self.c_chars = tf.placeholder(tf.int32, [None, self.config.context_len, self.config.max_char_len], 'context-chars')
        self.c_mask  = tf.sign(self.c_words)

        self.q_words = tf.placeholder(tf.int32, [None, self.config.question_len], 'query-words')
        self.q_chars = tf.placeholder(tf.int32, [None, self.config.question_len, self.config.max_char_len], 'query-chars')
        self.q_mask  = tf.sign(self.q_words)

        self.c_len = tf.cast(tf.reduce_sum(self.c_mask, -1), tf.int32)
        self.q_len = tf.cast(tf.reduce_sum(self.q_mask, -1), tf.int32)

        self.start = tf.placeholder(tf.int32, [None], 'start-index')
        self.end = tf.placeholder(tf.int32, [None], 'end-index')

        with tf.variable_scope('input-embedding'):
            c = tf.nn.embedding_lookup(self.word_embed, self.c_words)
            q = tf.nn.embedding_lookup(self.word_embed, self.q_words)

        with tf.variable_scope('rnn'):
            c = layers.birnn(c, self.c_len, self.config.cell_size, self.config.cell_type, self.dropout)
            q = layers.birnn(q, self.q_len, self.config.cell_size, self.config.cell_type, self.dropout, reuse=True)

        c_hats = [c]
        for i in range(self.aligner_hops):
            with tf.variable_scope('iterative-aligner-%d' % i):
                with tf.variable_scope('interactive-aligner'):
                    x = c_hats[-1]
                    similarity = tf.matmul(x, q, transpose_b=True)
                    _q = layers.bi_attention(x, q, similarity, self.c_mask, self.q_mask, only_c2q=True)
                    _c = layers.sfu(x, [_q, x * _q, x - _q])
                with tf.variable_scope('self-aligner'):
                    x = _c
                    similarity = tf.matmul(x, x, transpose_b=True)
                    _c = layers.bi_attention(x, x, similarity, self.c_mask, self.c_mask, only_c2q=True)
                    __c = layers.sfu(x, [_c, x * _c, x - _c])
                with tf.variable_scope('aggregate'):
                    x = __c
                    c_rnn = layers.birnn(x, self.c_len, self.config.cell_size, self.config.cell_type, self.dropout)
                    c_hats.append(c_rnn)

        def pointer(c, z, scope, reuse=None):
            with tf.variable_scope(scope, reuse=reuse):
                z = tf.tile(tf.expand_dims(z, 1), [1, self.config.context_len, 1]) # [batch, context_len, cell_size * 2]
                s = tf.concat([c, z, c * z], -1) # [batch, context_len, cell_size * 6]
                with tf.variable_scope('relu'):
                    s = tf.layers.dense(s, self.config.cell_size, activation=tf.nn.relu, reuse=reuse) # [batch, context_len, cell_size * 2]
                    s = tf.nn.dropout(s, 1.0 - self.dropout)
                with tf.variable_scope('linear'):
                    s = tf.squeeze(tf.layers.dense(s, 1, use_bias=False, reuse=reuse), -1) # [batch, context_len]
                    p = tf.nn.softmax(s - 1e30 * (1 - tf.cast(self.c_mask, tf.float32))) # [batch, context_len]
                return s, p

        def memory(c, p, z, scope, reuse=None):
            # c [batch, context_len, cell_size * 2] p [batch, context_len]
            with tf.variable_scope(scope, reuse=reuse):
                u = tf.squeeze(tf.matmul(c, tf.expand_dims(p, -1), transpose_a=True), -1) # [batch, cell_size * 2]
                return layers.sfu(z, [u], reuse=reuse)

        c_hat = c_hats[-1]
        z_s = q[:, -1, :]
        for i in range(self.pointer_hops):
            with tf.variable_scope('pointer-hop-%d' % i):
                start, p_start = pointer(c_hat, z_s, 'start-pointer')
                z_e = memory(c_hat, p_start, z_s, 'end-memory') # [batch, cell_size * 2]
                end, p_end = pointer(c_hat, z_e, 'end-pointer')

                if i + 1 < self.pointer_hops:
                    z_s = memory(c_hat, p_end, z_e, 'start-memory')

        self.start_linear = start
        self.end_linear = end

        self.pred_start = p_start
        self.pred_end = p_end

        with tf.variable_scope('loss') as scope:
            loss1 = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.start_linear, labels=self.start)
            loss2 = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.end_linear, labels=self.end)
            loss = tf.reduce_mean(loss1 + loss2)
            lossL2 = tf.add_n([ tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'bias' not in v.name ]) * self.config.l2
            self.loss = loss + lossL2

        with tf.variable_scope('optimizer') as scope:
            optimizer = AdamaxOptimizer(learning_rate=self.lr)
            grads = tf.gradients(self.loss, tf.trainable_variables())
            grads, _ = tf.clip_by_global_norm(grads, self.config.grad_clip)
            grads_and_vars = zip(grads, tf.trainable_variables())
            self.optimize = optimizer.apply_gradients(grads_and_vars, global_step=self.global_step)