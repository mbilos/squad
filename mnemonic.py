import tensorflow as tf
import numpy as np
import util

class MnemonicReader:
    def __init__(self, config):
        self.config = config

        self.global_step = tf.get_variable('global_step', shape=[], dtype=tf.int32, initializer=tf.constant_initializer(0), trainable=False)
        self.lr = tf.minimum(self.config.learning_rate, self.config.learning_rate / tf.log(999.) * tf.log(tf.cast(self.global_step, tf.float32) + 1))

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
            optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
            grads = tf.gradients(self.loss, tf.trainable_variables())
            grads, _ = tf.clip_by_global_norm(grads, self.config.grad_clip)
            grads_and_vars = zip(grads, tf.trainable_variables())
            self.optimize = optimizer.apply_gradients(grads_and_vars, global_step=self.global_step)

        with tf.variable_scope('ema') as scope:
            ema = tf.train.ExponentialMovingAverage(decay=self.config.ema_decay)
            ema_op = ema.apply(tf.trainable_variables())
            with tf.control_dependencies([ema_op]):
                self.loss = tf.identity(self.loss)
                assign_vars = []
                for var in tf.global_variables():
                    v = ema.average(var)
                    if v:
                        assign_vars.append(tf.assign(var,v))
            self.assign_vars = assign_vars

    def forward(self):
        self.c_char_embed, self.q_char_embed = self.char_embedding()
        self.c_encoded, self.q_encoded, self.q_state = self.input_encoder()
        self.modeling = self.iterative_aligner()
        self.start_linear, self.end_linear, self.pred_start, self.pred_end = self.answer_pointer()

    def answer_pointer(self):
        def pointer(c, z, scope):
            with tf.variable_scope(scope):
                z = tf.tile(tf.expand_dims(z, 1), [1, self.config.context_len, 1])
                s = tf.concat([c, z, c * z], -1)
                with tf.variable_scope('relu'):
                    s = tf.layers.dense(s, self.config.cell_size, activation=tf.nn.relu)
                    s = tf.layers.dropout(s, rate=self.config.dropout, training=self.config.training)
                with tf.variable_scope('linear'):
                    s = tf.squeeze(tf.layers.dense(s, 1, use_bias=False), -1)
                return s, tf.nn.softmax(s)

        def memory(c, p, z, scope):
            with tf.variable_scope(scope):
                u = tf.squeeze(tf.matmul(c, tf.expand_dims(p, -1), transpose_a=True), -1)
                return self.SFU(z, [u])

        def hop(c, z_s):
            start, p_start = pointer(c, z_s, 'start-pointer')
            z_e = memory(c, p_start, z_s, 'start-memory')

            end, p_end = pointer(c, z_e, 'end-pointer')
            z_s = memory(c, p_end, z_e, 'end-memory')

            return start, p_start, z_s, end, p_end, z_e

        start_memory = [self.q_state]

        for i in range(self.config.pointer_hops):
            with tf.variable_scope('pointer-hop-%d' % i):
                start, p_start, z_s, end, p_end, z_e = hop(self.modeling[-1], start_memory[-1])
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
            similarity = tf.matmul(c, c, transpose_b=True) - 1e30 * tf.eye(self.config.context_len)
            row_norm = tf.nn.softmax(similarity, -1)
            _c = tf.matmul(row_norm, c)
            return self.SFU(c, [_c, c * _c, c - _c])

    def interactive_aligning(self, c, q, scope='interactive-aligner', reuse=None):
        with tf.variable_scope(scope, reuse=reuse):
            similarity = tf.matmul(c, q, transpose_b=True)
            row_norm = tf.nn.softmax(similarity, -1)
            _q = tf.matmul(row_norm, q)
            return self.SFU(c, [_q, c * _q, c - _q])

    def SFU(self, inputs, fusion, scope='sfu', reuse=None):
        with tf.variable_scope(scope, reuse=reuse):
            shape = util.get_shape(inputs)
            x = tf.concat([inputs] + fusion, -1)

            res = tf.layers.dense(x, shape[-1], activation=tf.nn.tanh)
            res = tf.layers.dropout(res, rate=self.config.dropout, training=self.config.training)

            gate = tf.layers.dense(x, shape[-1], activation=tf.nn.sigmoid)
            gate = tf.layers.dropout(gate, rate=self.config.dropout, training=self.config.training)

            return gate * res + (1 - gate) * inputs

    def input_encoder(self):
        with tf.variable_scope('input-embedding'):
            similarity = tf.matmul(tf.nn.l2_normalize(self.c_words, -1), tf.nn.l2_normalize(self.q_words, -1), transpose_b=True)
            c_similarity = tf.reduce_max(similarity, -1, keep_dims=True)
            q_similarity = tf.transpose(tf.reduce_max(similarity, 1, keep_dims=True), [0,2,1])

            c = tf.concat([self.c_words, self.c_char_embed, c_similarity], -1)
            q = tf.concat([self.q_words, self.q_char_embed, q_similarity], -1)

            self.config.embed_size += 1

            if self.config.pos_embed > 0:
                self.pos_emb_matrix = tf.get_variable('pos_emb', [self.config.unique_pos, self.config.pos_embed])
                c_pos_embed = tf.nn.embedding_lookup(self.pos_emb_matrix, self.c_pos)
                q_pos_embed = tf.nn.embedding_lookup(self.pos_emb_matrix, self.q_pos)
                c = tf.concat([c, tf.layers.dropout(c_pos_embed, rate=self.config.dropout*0.5, training=self.config.training)], -1)
                q = tf.concat([q, tf.layers.dropout(q_pos_embed, rate=self.config.dropout*0.5, training=self.config.training)], -1)

            if self.config.ner_embed > 0:
                self.ner_emb_matrix = tf.get_variable('ner_emb', [self.config.unique_ner, self.config.ner_embed])
                c_ner_embed = tf.nn.embedding_lookup(self.ner_emb_matrix, self.c_ner)
                q_ner_embed = tf.nn.embedding_lookup(self.ner_emb_matrix, self.q_ner)
                c = tf.concat([c, tf.layers.dropout(c_ner_embed, rate=self.config.dropout*0.5, training=self.config.training)], -1)
                q = tf.concat([q, tf.layers.dropout(q_ner_embed, rate=self.config.dropout*0.5, training=self.config.training)], -1)

        with tf.variable_scope('contextual-embedding') as scope:
            c_output, _ = self.rnn(c, self.c_len)
            q_output, q_state = self.rnn(q, self.q_len, reuse=True)

        return c_output, q_output, q_state

    def char_embedding(self):
        with tf.variable_scope('char-embedding'):
            self.char_emb_matrix = tf.get_variable('char_emb', [self.config.unique_chars, self.config.char_embed])

            c = tf.nn.embedding_lookup(self.char_emb_matrix, self.c_chars)
            q = tf.nn.embedding_lookup(self.char_emb_matrix, self.q_chars)

            c = tf.layers.dropout(c, rate=self.config.dropout, training=self.config.training)
            q = tf.layers.dropout(q, rate=self.config.dropout, training=self.config.training)

            c = tf.layers.conv2d(c, self.config.char_embed, kernel_size=[1, 5], activation=tf.nn.relu)
            q = tf.layers.conv2d(q, self.config.char_embed, kernel_size=[1, 5], activation=tf.nn.relu, reuse=True)

            c = tf.reduce_max(c, axis=2)
            q = tf.reduce_max(q, axis=2)

            return c, q

    def rnn(self, inputs, sequence_length, reuse=None):
        return util.bidirectional_dynamic_rnn(
            inputs,
            sequence_length,
            self.config.cell_size,
            dropout=self.config.dropout,
            concat=True,
            reuse=reuse)