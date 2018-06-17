import tensorflow as tf
import numpy as np
import util

class QANet:
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
        self.c_encoded, self.q_encoded = self.input_encoder()
        self.attention = self.similarity()
        self.modeling = self.model_encoder()
        self.start_linear, self.end_linear, self.pred_start, self.pred_end = self.output()

    def encoder(self, inputs, num_blocks, num_convolutions, kernel, mask=None, scope='encoder', reuse=None):
        def layer_dropout(prev, residual, dropout):
            pred = tf.random_uniform([], 0.0, 1.0) < dropout
            return tf.cond(pred, lambda: prev, lambda: prev + residual)

        def residual_block(x, j):
            with tf.variable_scope('residual-block-%d' %j):
                ln = util.layer_norm(x, reuse=reuse)

                if (j + 1) % 2 == 0:
                    ln = tf.layers.dropout(ln, rate=self.config.dropout, training=self.config.training)

                conv = util.depthwise_separable_conv(
                    ln,
                    filters=self.config.filters,
                    activation=tf.nn.relu,
                    kernel_size=kernel,
                    dropout=self.config.dropout,
                    reuse=reuse,
                    training=self.config.training)

                return conv

        with tf.variable_scope(scope, reuse=reuse):
            block = [inputs]

            for i in range(num_blocks):
                with tf.variable_scope('encoder-block-%d' % i, reuse=reuse):
                    shape = util.get_shape(block[i])
                    pos = util.positional_encoding(block[i])

                    conv = [pos]
                    for j in range(num_convolutions):
                        res = residual_block(conv[j], j)
                        res = layer_dropout(conv[j], res, (j + 1) / num_convolutions * self.config.dropout)
                        conv.append(res)

                    with tf.variable_scope('self-attention'):
                        ln = util.layer_norm(conv[-1], reuse=reuse)
                        ln = tf.layers.dropout(ln, rate=self.config.dropout, training=self.config.training)

                        self_attention = util._multihead_attention(ln,
                            self.config.filters,
                            num_heads=self.config.num_heads,
                            reuse=reuse,
                            mask=mask,
                            is_training=self.config.training,
                            bias=False,
                            dropout=self.config.dropout)

                        self_attention = layer_dropout(conv[-1], self_attention, (i + 1) / num_blocks * self.config.dropout)

                    with tf.variable_scope('feedforward'):
                        ln = util.layer_norm(self_attention, reuse=reuse)
                        ff = tf.layers.dense(ln, shape[-1], activation=tf.nn.relu, reuse=reuse)
                        ff = tf.layers.dropout(ff, rate=self.config.dropout, training=self.config.training)

                    res = layer_dropout(self_attention, ff, (i + 1) / num_blocks * self.config.dropout)
                    block.append(res)

            return block[-1]

    def output(self):
        with tf.variable_scope('start-index') as scope:
            start = tf.concat([self.modeling[-3], self.modeling[-2]], -1)
            start = tf.squeeze(tf.layers.dense(start, 1, use_bias=False), -1)
            p_start = tf.nn.softmax(start, name='pred-start')

        with tf.variable_scope('end-index') as scope:
            end = tf.concat([self.modeling[-3], self.modeling[-1]], -1)
            end = tf.squeeze(tf.layers.dense(end, 1, use_bias=False), -1)
            p_end = tf.nn.softmax(end, name='pred-end')

        return start, end, p_start, p_end

    def model_encoder(self):
        with tf.variable_scope('modeling'):
            model_input = tf.layers.conv1d(self.attention, self.config.filters, 1, padding='same')
            modeling = [model_input]

            for i in range(3):
                if i % 2 == 0:
                    modeling[i] = tf.layers.dropout(modeling[i], rate=self.config.dropout, training=self.config.training)
                reuse = True if i > 0 else None
                m = self.encoder(
                    modeling[i],
                    num_blocks=self.config.model_num_blocks,
                    num_convolutions=self.config.model_num_convs,
                    kernel=self.config.model_kernel,
                    mask=self.c_mask,
                    reuse=reuse)
                modeling.append(m)

            return modeling

    def similarity(self):
        with tf.variable_scope('attention'):
            c, q = self.c_encoded, self.q_encoded

            c_tile = tf.tile(tf.expand_dims(c, 2), [1, 1, self.config.question_len, 1])
            q_tile = tf.tile(tf.expand_dims(q, 1), [1, self.config.context_len, 1, 1])

            similarity = tf.concat([c_tile, q_tile, c_tile * q_tile], -1)
            similarity = tf.squeeze(tf.layers.dense(similarity, 1, use_bias=False), -1)

            row_norm = tf.nn.softmax(similarity)
            A = tf.matmul(row_norm, q) # context to query

            column_norm = tf.nn.softmax(similarity, 1)
            B = tf.matmul(tf.matmul(row_norm, column_norm, transpose_b=True), c) # query to context

            attention = tf.concat([c, A, c * A, c * B], -1)

            return attention

    def input_encoder(self):
        with tf.variable_scope('input-encoder'):
            c, q = self.input_embedding()

            c = self.encoder(c, self.config.encoder_num_blocks, self.config.encoder_num_convs, self.config.encoder_kernel, mask=self.c_mask)
            q = self.encoder(q, self.config.encoder_num_blocks, self.config.encoder_num_convs, self.config.encoder_kernel, mask=self.q_mask, reuse=True)

            return c, q

    def input_embedding(self):
        with tf.variable_scope('input-embedding'):
            c_char_embed, q_char_embed = self.char_embedding()

            similarity = tf.matmul(tf.nn.l2_normalize(self.c_words, -1), tf.nn.l2_normalize(self.q_words, -1), transpose_b=True)
            c_similarity = tf.reduce_max(similarity, -1, keep_dims=True)
            q_similarity = tf.transpose(tf.reduce_max(similarity, 1, keep_dims=True), [0,2,1])

            c = tf.concat([self.c_words, c_char_embed, c_similarity], -1)
            q = tf.concat([self.q_words, q_char_embed, q_similarity], -1)

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

            with tf.variable_scope('highway'):
                with tf.variable_scope('highway-1'):
                    c_h1 = tf.layers.dense(c, self.config.embed_size, activation=tf.nn.relu)
                    c_h1 = tf.layers.dropout(c_h1, rate=self.config.dropout, training=self.config.training)
                    c_h1 = util.gated_connection(c, c_h1)

                    q_h1 = tf.layers.dense(q, self.config.embed_size, activation=tf.nn.relu, reuse=True)
                    q_h1 = tf.layers.dropout(q_h1, rate=self.config.dropout, training=self.config.training)
                    q_h1 = util.gated_connection(q, q_h1, reuse=True)

                with tf.variable_scope('highway-2'):
                    c_h2 = tf.layers.dense(c_h1, self.config.embed_size, activation=tf.nn.relu)
                    c_h2 = tf.layers.dropout(c_h2, rate=self.config.dropout, training=self.config.training)
                    c_h2 = util.gated_connection(c_h1, c_h2)

                    q_h2 = tf.layers.dense(q_h1, self.config.embed_size, activation=tf.nn.relu, reuse=True)
                    q_h2 = tf.layers.dropout(q_h2, rate=self.config.dropout, training=self.config.training)
                    q_h2 = util.gated_connection(q_h1, q_h2, reuse=True)

                c_conv = tf.layers.conv1d(c_h2, self.config.filters, 1, padding='same')
                q_conv = tf.layers.conv1d(q_h2, self.config.filters, 1, padding='same', reuse=True)

            return c_conv, q_conv

    def char_embedding(self):
         with tf.variable_scope('char-embedding'):
            self.char_emb_matrix = tf.get_variable('char_emb', [self.config.unique_chars, self.config.char_embed])

            c = tf.nn.embedding_lookup(self.char_emb_matrix, self.c_chars)
            q = tf.nn.embedding_lookup(self.char_emb_matrix, self.q_chars)

            c = tf.layers.conv2d(c, self.config.char_embed, kernel_size=[1, 5], activation=tf.nn.relu)
            q = tf.layers.conv2d(q, self.config.char_embed, kernel_size=[1, 5], activation=tf.nn.relu, reuse=True)

            c = tf.reduce_max(tf.layers.dropout(c, rate=self.config.dropout * 0.5, training=self.config.training), axis = 2)
            q = tf.reduce_max(tf.layers.dropout(q, rate=self.config.dropout * 0.5, training=self.config.training), axis = 2)

            return c, q