import tensorflow as tf
import numpy as np
import util

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

        self.global_step = tf.get_variable('global_step', shape=[], dtype=tf.int32, initializer=tf.constant_initializer(0), trainable=False)
        self.lr = tf.get_variable('learning-rate', shape=[], dtype=tf.float32, initializer=tf.constant_initializer(self.config.learning_rate), trainable=False)
        self.decay_lr = tf.assign(self.lr, tf.maximum(self.lr / 2, 1e-6))

        self.input()
        self.forward()
        self.training()

    def input(self):
        with tf.variable_scope('input') as scope:
            self.c_words = tf.placeholder(tf.float32, [None, self.config.context_len, self.config.word_embed], 'context-words')
            self.c_chars = tf.placeholder(tf.int32, [None, self.config.context_len, self.config.max_char_len], 'context-chars')
            self.c_pos = tf.placeholder(tf.int32, [None, self.config.context_len], 'context-part-of-speech')
            self.c_ner = tf.placeholder(tf.int32, [None, self.config.context_len], 'context-named-entity')

            self.q_words = tf.placeholder(tf.float32, [None, self.config.question_len, self.config.word_embed], 'query-words')
            self.q_chars = tf.placeholder(tf.int32, [None, self.config.question_len, self.config.max_char_len], 'query-chars')
            self.q_pos = tf.placeholder(tf.int32, [None, self.config.question_len], 'query-part-of-speech')
            self.q_ner = tf.placeholder(tf.int32, [None, self.config.question_len], 'query-named-entity')

            self.c_mask = tf.cast(tf.cast(tf.reduce_sum(self.c_words, -1), tf.bool), tf.float32)
            self.q_mask = tf.cast(tf.cast(tf.reduce_sum(self.q_words, -1), tf.bool), tf.float32)

            self.c_len = tf.cast(tf.reduce_sum(self.c_mask, -1), tf.int32)
            self.q_len = tf.cast(tf.reduce_sum(self.q_mask, -1), tf.int32)

            self.start = tf.placeholder(tf.int32, [None], 'start-index')
            self.end = tf.placeholder(tf.int32, [None], 'end-index')

    def training(self):
        with tf.variable_scope('loss') as scope:
            loss1 = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.start_linear, labels=self.start)
            loss2 = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.end_linear, labels=self.end)
            loss = tf.reduce_mean(loss1 + loss2)
            lossL2 = tf.add_n([ tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'bias' not in v.name ]) * self.config.l2
            self.loss = loss + lossL2

        with tf.variable_scope('optimizer') as scope:
            optimizer = tf.train.AdamOptimizer()
            grads = tf.gradients(self.loss, tf.trainable_variables())
            grads, _ = tf.clip_by_global_norm(grads, self.config.grad_clip)
            grads_and_vars = zip(grads, tf.trainable_variables())
            self.optimize = optimizer.apply_gradients(grads_and_vars, global_step=self.global_step)

    def forward(self):
        self.c_encoded, self.q_encoded = self.input_encoder()
        self.modeling = self.iterative_aligner()
        self.start_linear, self.end_linear, self.pred_start, self.pred_end = self.answer_pointer()

    def answer_pointer(self):
        def pointer(c, z, scope):
            with tf.variable_scope(scope):
                z = tf.tile(tf.expand_dims(z, 1), [1, self.config.context_len, 1])
                s = tf.concat([c, z, c * z], -1)
                with tf.variable_scope('relu'):
                    s = tf.layers.dense(s, self.config.cell_size * 6, activation=tf.nn.relu)
                    s = tf.layers.dropout(s, rate=self.config.dropout, training=self.config.training)
                with tf.variable_scope('linear'):
                    s = tf.squeeze(tf.layers.dense(s, 1, use_bias=False), -1)
                    p = tf.nn.softmax(s - 1e30 * (1 - self.c_mask))
                return s, p

        def memory(c, p, z, scope):
            with tf.variable_scope(scope):
                u = tf.squeeze(tf.matmul(c, tf.expand_dims(p, -1), transpose_a=True), -1)
                return self.SFU(z, [u])

        def hop(c, z_s, start_memory=True):
            start, p_start = pointer(c, z_s, 'start-pointer')

            z_e = memory(c, p_start, z_s, 'end-memory')
            end, p_end = pointer(c, z_e, 'end-pointer')

            z_s = memory(c, p_end, z_e, 'start-memory') if start_memory else None

            return start, p_start, z_s, end, p_end, z_e

        start_memory = [self.q_encoded[:,-1,:]]

        for i in range(self.config.pointer_hops):
            with tf.variable_scope('pointer-hop-%d' % i):
                calc_start = i + 1 < self.config.pointer_hops
                start, p_start, z_s, end, p_end, z_e = hop(self.modeling[-1], start_memory[-1], calc_start)
                start_memory.append(z_s)

        return start, end, p_start, p_end

    def iterative_aligner(self):
        def hop(c, q):
            _c = self.interactive_aligning(c, q)
            __c = self.self_aligning(_c)
            out, _ = self.rnn(__c, self.c_len)
            return out

        c_hats = [self.c_encoded]
        for i in range(self.config.aligner_hops):
            with tf.variable_scope('iterative-hop-%d' % i):
                c_hats.append(hop(c_hats[-1], self.q_encoded))
        return c_hats

    def self_aligning(self, c,  scope='self-aligner', reuse=None):
        with tf.variable_scope(scope, reuse=reuse):
            shape = util.get_shape(c)
            proj = tf.layers.dense(c, shape[-1], activation=tf.nn.relu)

            similarity = tf.matmul(proj, proj, transpose_b=True) - 1e30 * tf.eye(self.config.context_len)
            similarity -= 1e30 * (1 - tf.expand_dims(self.c_mask, 1))
            row_norm = tf.nn.softmax(similarity, -1)
            _c = tf.matmul(row_norm, c)

            return self.SFU(c, [_c, c * _c, c - _c])

    def interactive_aligning(self, c, q, scope='interactive-aligner', reuse=None):
        with tf.variable_scope(scope, reuse=reuse):
            similarity = tf.matmul(c, q, transpose_b=True)
            similarity -= 1e30 * (1 - tf.expand_dims(self.q_mask, 1))
            row_norm = tf.nn.softmax(similarity, -1)
            _q = tf.matmul(row_norm, q)
            return self.SFU(c, [_q, c * _q, c - _q])

    def SFU(self, inputs, fusion, scope='sfu', reuse=None):
        with tf.variable_scope(scope, reuse=reuse):
            shape = util.get_shape(inputs)
            x = tf.concat([inputs] + fusion, -1)

            res = tf.layers.dense(x, shape[-1], activation=tf.nn.tanh)
            gate = tf.layers.dense(x, shape[-1], activation=tf.nn.sigmoid)

            return gate * res + (1 - gate) * inputs

    def input_encoder(self):
        with tf.variable_scope('input-embedding'):
            c = self.c_words
            q = self.q_words

            self.config.embed_size = self.config.word_embed

        with tf.variable_scope('contextual-embedding') as scope:
            c_output, _ = self.rnn(c, self.c_len)
            q_output, _ = self.rnn(q, self.q_len, reuse=True)

        return c_output, q_output

    def rnn(self, inputs, sequence_length, reuse=None):
        return util.bidirectional_dynamic_rnn(
            inputs,
            sequence_length,
            self.config.cell_size,
            dropout=self.config.dropout,
            concat=True,
            reuse=reuse)