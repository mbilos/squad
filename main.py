import os
import argparse
import numpy as np
import tensorflow as tf
from tqdm import tqdm

import read
import evaluate

from bidaf import BiDAF
from mnemonic import MnemonicReader

class Main:
    def __init__(self):
        self.config = self.get_args()
        self.trainset, self.devset, self.config.embed, self.word2index, self.char2index = read.data()

        self.config.unique_chars = len(self.char2index)

        with tf.Graph().as_default() as g:
            if self.config.name == 'bidaf':
                self.model = BiDAF(self.config)
            elif self.config.name == 'mnemonic':
                self.model = MnemonicReader(self.config)
            else:
                raise NotImplementedError('Invalid arhitecture name')

            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                saver = tf.train.Saver(max_to_keep=20)
                save_path = os.path.join('models', self.config.name)

                if os.path.exists(save_path):
                    saver.restore(sess, tf.train.latest_checkpoint(save_path))

                if self.config.mode == 'train':
                    self.train(sess, saver)
                else:
                    step = sess.run(self.model.global_step)

                    em, f1 = self.test(sess)
                    print('\nIteration: %d - Exact match: %.2f\tf1: %.2f\t' % (step, em, f1))

                    if self.config.ema_decay > 0:
                        sess.run(self.model.assign_vars)
                        ema, ema_f1 = self.test(sess)
                        print('\nIteration EMA: %d - Exact match: %.2f\tf1: %.2f' % (step, ema, ema_f1))

    def train(self, sess, saver):
        best_f1 = 0
        loss = 0
        step = sess.run(self.model.global_step)

        t = tqdm(range(step, self.config.iterations))
        for i in t:
            c, ch, q, qh, s, e = self.get_batch('train')
            feed = { self.model.c_words: c, self.model.c_chars: ch, self.model.q_words: q,
                     self.model.q_chars: qh, self.model.start: s, self.model.end: e }

            _, loss = sess.run([self.model.optimize, self.model.loss], feed)
            t.set_description('loss: %.2f' % loss)

            if i > 0 and i % self.config.save_every == 0:
                saver.save(sess, os.path.join('models', self.config.name, 'model'), global_step=i)

                em, f1 = self.test(sess)
                print('\nIteration: %d - Exact match: %.2f\tf1: %.2f\tlr: %f' % (i, em, f1, sess.run(self.model.lr)))

                if i % 5000 == 0 and self.config.ema_decay > 0:
                    sess.run(self.model.assign_vars)
                    ema, ema_f1 = self.test(sess)
                    print('Iteration EMA: %d - Exact match: %.2f\tf1: %.2f' % (i, ema, ema_f1))

                if f1 > best_f1:
                    best_f1 = f1
                else:
                    sess.run(self.model.decay_lr)
                    print('best f1: %.2f - current f1: %.2f - new lr: %f' % (best_f1, f1, sess.run(self.model.lr)))

    def test(self, sess):
        total = em = f1 = 0
        for i in range(0, len(self.devset), 50):
            tokens, c, ch, q, qh, answers = self.get_batch('test', i, i + 50)
            feed = { self.model.c_words: c, self.model.c_chars: ch, self.model.q_words: q, self.model.q_chars: qh }

            start, end = sess.run([self.model.pred_start, self.model.pred_end], feed)
            start, end = self.get_best_spans(start, end)

            answer_cand = [' '.join(x[k:l]) for x,k,l in zip(tokens, start, end)]

            e, f = evaluate._evaluate(answers, answer_cand)
            em += e
            f1 += f
            total += 1

        return em / total, f1 / total

    def get_best_spans(self, start, end):
        def get_best_span(first, second):
            max_val = 0
            best_word_span = (0, 1)
            argmax_j1 = 0
            for j in range(len(first)):
                val1 = first[argmax_j1]
                if val1 < first[j]:
                    val1 = first[j]
                    argmax_j1 = j

                val2 = second[j]
                if val1 * val2 > max_val and j != argmax_j1 and argmax_j1 - j < self.config.answer_len:
                    best_word_span = (argmax_j1, j)
                    max_val = val1 * val2
            return best_word_span

        starts = []
        ends = []
        for x, y in zip(start, end):
            s, e = get_best_span(x, y)
            starts.append(s)
            ends.append(e)
        return np.array(starts), np.array(ends)

    def get_batch(self, mode='train', start=None, end=None):
        PAD = 'PAD'
        UNK = 'UNK'

        data = self.trainset if mode == 'train' else self.devset

        if mode == 'test':
            batch = data[start:end]
        else:
            indexes = np.random.randint(len(data), size=self.config.batch)
            batch = [data[i] for i in indexes]

        def _embedding(w):
            if w in self.word2index:
                return self.word2index[w]
            elif w.lower() in self.word2index:
                return self.word2index[w.lower()]
            else:
                return self.word2index[UNK]

        tokens, c_words, c_chars, q_words, q_chars, start, end, answers = [], [], [], [], [], [], [], []

        for b in batch:
            c_tokens = b[4][:self.config.context_len]
            q_tokens = b[7][:self.config.question_len]

            c_pad = self.config.context_len - len(c_tokens)
            q_pad = self.config.question_len - len(q_tokens)

            c_words.append([_embedding(x) for x in c_tokens + tuple([PAD]) * c_pad])
            q_words.append([_embedding(x) for x in q_tokens + tuple([PAD]) * q_pad])

            cch = [list(x)[:self.config.max_char_len] + [PAD] * (self.config.max_char_len - len(list(x))) for x in c_tokens + tuple(['']) * c_pad]
            c_chars.append([[self.char2index.get(x, self.char2index[UNK]) for x in s] for s in cch])

            qch = [list(x)[:self.config.max_char_len] + [PAD] * (self.config.max_char_len - len(list(x))) for x in q_tokens + tuple(['']) * q_pad]
            q_chars.append([[self.char2index.get(x, self.char2index[UNK]) for x in s] for s in qch])

            if mode == 'train':
                start.append(min(b[-2], self.config.context_len - 1))
                end.append(min(b[-1], self.config.question_len - 1))
            else:
                tokens.append(c_tokens)
                answers.append(b[3])

        c_words = np.array(c_words)
        c_chars = np.array(c_chars)
        q_words = np.array(q_words)
        q_chars = np.array(q_chars)
        start = np.array(start)
        end = np.array(end)

        if mode == 'train':
            return c_words, c_chars, q_words, q_chars, start, end
        else:
            return tokens, c_words, c_chars, q_words, q_chars, answers

    def get_args(self):
        parser = argparse.ArgumentParser(description='')
        parser.add_argument('--name',           required=True,      type=str)
        parser.add_argument('--mode',           default = 'train',  type=str)
        parser.add_argument('--batch',          default = 32,       type=int)
        parser.add_argument('--save_every',     default = 1000,     type=int)
        parser.add_argument('--iterations',     default = 30001,    type=int)
        parser.add_argument('--context_len',    default = 400,      type=int)
        parser.add_argument('--question_len',   default = 30,       type=int)
        parser.add_argument('--answer_len',     default = 15,       type=int)
        parser.add_argument('--char_embed',     default = 64,       type=int)
        parser.add_argument('--max_char_len',   default = 16,       type=int)
        parser.add_argument('--learning_rate',  default = 0.001,    type=float)
        parser.add_argument('--filters',        default = 128,      type=int)
        parser.add_argument('--dropout',        default = 0.1,      type=float)
        parser.add_argument('--l2',             default = 3e-7,     type=float)
        parser.add_argument('--grad_clip',      default = 5.0,      type=float)
        parser.add_argument('--ema_decay',      default = 0.999,    type=float)
        parser.add_argument('--cell_size',      default = 128,      type=int)
        parser.add_argument('--cell_type',      default = 'gru',    type=str)

        args = parser.parse_args()
        args.training = args.mode == 'train'

        if not args.training:
            args.dropout = 0.0

        for a in vars(args):
            print('{:<20}'.format(a), getattr(args, a))

        args.embed_size = 300 + args.char_embed

        return args

if __name__ == "__main__":
    Main()