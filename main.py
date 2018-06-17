import os
import argparse

import tensorflow as tf
import numpy as np
from tqdm import tqdm

import read
import evaluate
from qanet import QANet
from bidaf import BiDAF
from bidaf_self_attention import BiDAF_SelfAttention
from bidaf_conv_input import BiDAF_ConvInput
from mnemonic import MnemonicReader

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class Main:
    def __init__(self):
        self.config = self.get_args()

        self.trainset, self.devset, self.embed, self.char2index, \
            self.index2char, self.tag2index, self.index2tag, \
            self.entity2index, self.index2entity = read.data(self.config.word_embed)

        self.config.unique_chars = len(self.char2index)
        self.config.unique_pos = len(self.tag2index)
        self.config.unique_ner = len(self.entity2index)
        self.config.embed_size = self.config.word_embed + self.config.char_embed + \
            self.config.pos_embed + self.config.ner_embed

        with tf.Graph().as_default() as g:
            if self.config.name == 'qanet':
                self.model = QANet(self.config)
            elif self.config.name == 'bidaf':
                self.model = BiDAF(self.config)
            elif self.config.name == 'bidaf-att':
                self.model = BiDAF_SelfAttention(self.config)
            elif self.config.name == 'bidaf-conv-input':
                self.model = BiDAF_ConvInput(self.config)
            elif self.config.name == 'mnemonic':
                self.model = MnemonicReader(self.config)
            else:
                raise NotImplementedError('Invalid arhitecture name')

            if self.config.mode == 'train':
                self.train()
            else:
                self.test()

    def train(self):
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver(max_to_keep=20)

            save_path = os.path.join('models', self.config.name)
            if os.path.exists(save_path):
                saver.restore(sess, tf.train.latest_checkpoint(save_path))

            best_f1 = 0
            loss = 0
            step = sess.run(self.model.global_step)

            t = tqdm(range(step, self.config.iterations))
            for i in t:
                c, ch, q, qh, ct, ce, qt, qe, s, e = self.get_batch('train')
                feed = { self.model.c_words: c, self.model.c_chars: ch, self.model.c_pos: ct, self.model.c_ner: ce,
                         self.model.q_words: q, self.model.q_chars: qh, self.model.q_pos: qt, self.model.q_ner: qe,
                         self.model.start: s, self.model.end: e }

                _, loss = sess.run([self.model.optimize, self.model.loss], feed)
                t.set_description('loss: %.2f' % loss)

                if i > 0 and i % self.config.save_every == 0:
                    saver.save(sess, os.path.join('models', self.config.name, 'model'), global_step=i)

                    em, f1 = self.test(sess)
                    print('\nIteration: %d - Exact match: %.2f\tf1: %.2f\tlr: %f' % (i, em, f1, sess.run(self.model.lr)))

                    if i % 5000 == 0 and self.config.ema_decay > 0:
                        sess.run(model.assign_vars)
                        ema, ema_f1 = test(sess)
                        print('\nIteration EMA: %d - Exact match: %.2f\tf1: %.2f' % (i, ema, ema_f1))

                    if f1 > best_f1:
                        best_f1 = f1
                    else:
                        sess.run(self.model.decay_lr)
                        print('best f1: %.2f - current f1: %.2f - new lr: %f' % (best_f1, f1, sess.run(self.model.lr)))

    def test(self, sess):
        total = em = f1 = 0
        for i in range(0, len(self.devset), 50):
            tokens, c, ch, q, qh, ct, ce, qt, qe, answers = self.get_batch('test', i, i + 50)
            feed = { self.model.c_words: c, self.model.c_chars: ch, self.model.c_pos: ct, self.model.c_ner: ce,
                    self.model.q_words: q, self.model.q_chars: qh, self.model.q_pos: qt, self.model.q_ner: qe }

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
        PAD = '='
        UNK = '_'

        data = self.trainset if mode == 'train' else self.devset

        if mode == 'test':
            batch = data[start:end]
        else:
            indexes = np.random.randint(len(data), size=self.config.batch)
            batch = [data[i] for i in indexes]

        def _embedding(w):
            if w in self.embed:
                return self.embed[w]
            elif w.lower() in self.embed:
                return self.embed[w.lower()]
            else:
                return self.embed[UNK]

        def add_padding(data, dictionary, length, max_length):
            return [dictionary[x] for x in data][:max_length] + [dictionary[UNK]] * (max_length - length)

        max_ch = self.config.max_char_len
        unk_ch_ind = self.char2index[UNK]
        contexts, context_ch, questions, question_ch = [], [], [], []
        context_tags, context_entities, question_tags, question_entities = [], [], [], []
        for b in batch:
            c_length = len(b[4])
            q_length = len(b[7])
            context = b[4][:self.config.context_len] + tuple(PAD) * (self.config.context_len - c_length)
            question = b[7][:self.config.question_len] + tuple(PAD) * (self.config.question_len - q_length)
            contexts.append([_embedding(x) for x in context])
            questions.append([_embedding(x) for x in question])
            context_ch.append([[self.char2index.get(c, unk_ch_ind) for c in list(x[:max_ch] + PAD * (max_ch - len(x)))] for x in context])
            question_ch.append([[self.char2index.get(c, unk_ch_ind) for c in list(x[:max_ch] + PAD * (max_ch - len(x)))] for x in question])
            context_tags.append(add_padding(b[5], self.tag2index, c_length, self.config.context_len))
            context_entities.append(add_padding(b[6], self.entity2index, c_length, self.config.context_len))
            question_tags.append(add_padding(b[8], self.tag2index, q_length, self.config.question_len))
            question_entities.append(add_padding(b[9], self.entity2index, q_length, self.config.question_len))

        if mode == 'train':
            starts = [min(x[-2], self.config.context_len - 1) for x in batch]
            ends = [min(x[-1], self.config.context_len - 1) for x in batch]

            return contexts, context_ch, questions, question_ch, context_tags, \
                context_entities, question_tags, question_entities, np.array(starts), np.array(ends)
        else:
            tokens = [x[4] for x in batch]
            answers = [x[3] for x in batch]
            return tokens, contexts, context_ch, questions, question_ch, context_tags, \
                context_entities, question_tags, question_entities, answers

    def get_args(self):
        parser = argparse.ArgumentParser(description='')
        parser.add_argument('--name',           required=True,      type=str)
        parser.add_argument('--mode',           default = 'train',  type=str)
        parser.add_argument('--print_every',    default = 50,       type=int)
        parser.add_argument('--batch',          default = 32,       type=int)
        parser.add_argument('--save_every',     default = 1000,     type=int)
        parser.add_argument('--iterations',     default = 30001,    type=int)
        parser.add_argument('--context_len',    default = 400,      type=int)
        parser.add_argument('--question_len',   default = 30,       type=int)
        parser.add_argument('--answer_len',     default = 15,       type=int)
        parser.add_argument('--word_embed',     default = 300,      type=int)
        parser.add_argument('--char_embed',     default = 200,      type=int)
        parser.add_argument('--pos_embed',      default = 20,       type=int)
        parser.add_argument('--ner_embed',      default = 20,       type=int)
        parser.add_argument('--max_char_len',   default = 16,       type=int)
        parser.add_argument('--learning_rate',  default = 0.001,    type=float)
        parser.add_argument('--filters',        default = 128,      type=int)
        parser.add_argument('--dropout',        default = 0.1,      type=float)
        parser.add_argument('--l2',             default = 3e-7,     type=float)
        parser.add_argument('--grad_clip',      default = 5.0,      type=float)
        parser.add_argument('--ema_decay',      default = 0.9999,   type=float)

        # qanet specific
        parser.add_argument('--encoder_num_blocks', default = 1, type=int)
        parser.add_argument('--encoder_num_convs',  default = 4, type=int)
        parser.add_argument('--encoder_kernel',     default = 7, type=int)
        parser.add_argument('--model_num_blocks',   default = 7, type=int)
        parser.add_argument('--model_num_convs',    default = 2, type=int)
        parser.add_argument('--model_kernel',       default = 5, type=int)
        parser.add_argument('--passes',             default = 3, type=int)
        parser.add_argument('--num_heads',          default = 8, type=int)

        # bidaf specific
        parser.add_argument('--cell_size', default = 128, type=int)

        # mnemonic specific
        parser.add_argument('--aligner_hops', default = 2, type=int)
        parser.add_argument('--pointer_hops', default = 2, type=int)

        args = parser.parse_args()
        args.training = args.mode == 'train'

        for a in vars(args):
            print('{:<20}'.format(a), getattr(args, a))

        return args

if __name__ == "__main__":
    Main()