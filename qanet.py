import tensorflow as tf
import numpy as np
import util

class QANet:
    def __init__(self, config):
        self.config = config

        self.global_step = tf.get_variable('global_step', shape=[], dtype=tf.int32, initializer=tf.constant_initializer(0), trainable=False)
        self.lr = tf.minimum(self.config.learning_rate, self.config.learning_rate / tf.log(999.) * tf.log(tf.cast(self.global_step, tf.float32) + 1))

        self.input()
        self.forward()
        self.training()

    def input(self):
        self.c_words = tf.placeholder(tf.float32, [None, self.config.context_len, self.config.word_embed], 'context-words')
        self.c_chars = tf.placeholder(tf.int32, [None, self.config.context_len, self.config.max_char_len], 'context-chars')

        self.q_words = tf.placeholder(tf.float32, [None, self.config.question_len, self.config.word_embed], 'query-words')
        self.q_chars = tf.placeholder(tf.int32, [None, self.config.question_len, self.config.max_char_len], 'query-chars')

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
                loss = tf.identity(self.loss)
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

    def encoder(self, inputs, num_blocks, num_convolutions, kernel, scope='encoder', reuse=None):
        with tf.variable_scope(scope, reuse=reuse):
            output = inputs

            for i in range(num_blocks):
                with tf.variable_scope('encoder-block-' + str(i), reuse=reuse):
                    shape = output.get_shape().as_list()
                    conv_input = util.positional_encoding(output, dropout=self.config.dropout)

                    def residual_block(x, j):
                        with tf.variable_scope('residual-block-%d' %j):
                            ln = util.layer_norm(x, scope='layer-norm-%d' %j, reuse=reuse)
                            conv = util.depthwise_separable_conv(ln, self.config.filters, kernel, reuse=reuse)
                            return x + conv

                    for j in range(num_convolutions):
                        conv_input = residual_block(conv_input, j)

                    with tf.variable_scope('self-attention'):
                        conv_ln = tf.nn.dropout(util.layer_norm(conv_input, scope='layer-norm-%d-1' %j, reuse=reuse), 1 - self.config.dropout)
                        self_attention = util.multihead_attention(conv_ln, conv_ln, conv_ln, heads=self.config.num_heads, scope='self-attention-%d-1' %j, reuse=reuse)
                        self_attention = tf.nn.dropout(self_attention, 1 - self.config.dropout) + conv_input

                    with tf.variable_scope('feedforward'):
                        att_ln = util.layer_norm(self_attention, scope='layer-norm-%d-2' %j, reuse=reuse)
                        feedforward = util.dense(att_ln, shape[-1], activation=tf.nn.relu, scope='ff_1', reuse=reuse)
                        feedforward = util.dense(feedforward, shape[-1], activation=None, scope='ff_2', reuse=reuse, dropout=self.config.dropout)

                    output = feedforward + self_attention

        return output

    def output(self):
        with tf.variable_scope('start-index') as scope:
            start = tf.concat([self.modeling[-3], self.modeling[-2]], -1)
            start = tf.squeeze(util.dense(start, 1), -1)
            p_start = tf.nn.softmax(start, name='pred-start')

        with tf.variable_scope('end-index') as scope:
            end = tf.concat([self.modeling[-3], self.modeling[-1]], -1)
            end = tf.squeeze(util.dense(end, 1), -1)
            p_end = tf.nn.softmax(end, name='pred-end')

        return start, end, p_start, p_end

    def model_encoder(self):
        with tf.variable_scope('modeling'):
            model_input = tf.layers.conv1d(self.attention, self.config.filters, 1, padding='same')
            modeling = [model_input]

            for i in range(3):
                if i % 2 == 0:
                    modeling[i] = tf.nn.dropout(modeling[i], 1.0 - self.config.dropout)
                reuse = True if i > 0 else None
                modeling.append(self.encoder(modeling[i], self.config.model_num_blocks,
                    self.config.model_num_convs, self.config.model_kernel, reuse=reuse))

            return modeling

    def similarity(self):
        with tf.variable_scope('attention'):
            c, q = self.c_encoded, self.q_encoded

            c_tile = tf.tile(tf.expand_dims(c, 2), [1, 1, self.config.question_len, 1])
            q_tile = tf.tile(tf.expand_dims(q, 1), [1, self.config.context_len, 1, 1])

            similarity = tf.concat([c_tile, q_tile, c_tile * q_tile], -1)
            similarity = tf.squeeze(util.dense(similarity, 1), -1)

            row_norm = tf.nn.softmax(similarity)
            A = tf.matmul(row_norm, q) # context to query

            column_norm = tf.nn.softmax(similarity, 1)
            B = tf.matmul(tf.matmul(row_norm, column_norm, transpose_b=True), c) # query to context

            attention = tf.concat([c, A, c * A, c * B], -1)

            return attention

    def input_encoder(self):
        with tf.variable_scope('input-encoder'):
            c, q = self.input_embedding()

            c = self.encoder(c, self.config.encoder_num_blocks, self.config.encoder_num_convs, self.config.encoder_kernel)
            q = self.encoder(q, self.config.encoder_num_blocks, self.config.encoder_num_convs, self.config.encoder_kernel, reuse=True)

            return c, q

    def input_embedding(self):
        with tf.variable_scope('input-embedding'):
            c_char_embed, q_char_embed = self.char_embedding()

            c = tf.concat([self.c_words, c_char_embed], -1)
            q = tf.concat([self.q_words, q_char_embed], -1)

            with tf.variable_scope('highway'):
                with tf.variable_scope('highway-1'):
                    c_h1 = util.dense(c, self.config.embed_size, activation=tf.nn.relu, dropout=self.config.dropout)
                    c_h1 = util.gated_connection(c, c_h1)

                    q_h1 = util.dense(q, self.config.embed_size, activation=tf.nn.relu, dropout=self.config.dropout, reuse=True)
                    q_h1 = util.gated_connection(q, q_h1, reuse=True)

                with tf.variable_scope('highway-2'):
                    c_h2 = util.dense(c_h1, self.config.embed_size, activation=tf.nn.relu, dropout=self.config.dropout)
                    c_h2 = util.gated_connection(c_h1, c_h2)

                    q_h2 = util.dense(q_h1, self.config.embed_size, activation=tf.nn.relu, dropout=self.config.dropout, reuse=True)
                    q_h2 = util.gated_connection(q_h1, q_h2, reuse=True)

                c_conv = tf.layers.conv1d(c_h2, self.config.filters, 1, padding='same')
                q_conv = tf.layers.conv1d(q_h2, self.config.filters, 1, padding='same', reuse=True)

            return c_conv, q_conv

    def char_embedding(self):
         with tf.variable_scope('char-embedding'):
            self.char_emb_matrix = tf.get_variable('char_emb', [self.config.unique_chars, self.config.char_embed])
            char_keep = 1.0 - 0.5 * self.config.dropout

            c = tf.nn.embedding_lookup(self.char_emb_matrix, self.c_chars)
            q = tf.nn.embedding_lookup(self.char_emb_matrix, self.q_chars)

            c = tf.reduce_max(tf.nn.dropout(c, char_keep), axis = 2)
            q = tf.reduce_max(tf.nn.dropout(q, char_keep), axis = 2)

            c = tf.layers.conv1d(c, self.config.char_embed, kernel_size=5, activation=tf.nn.relu, padding='same')
            q = tf.layers.conv1d(q, self.config.char_embed, kernel_size=5, activation=tf.nn.relu, padding='same', reuse=True)

            return c, q