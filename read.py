import os
import numpy as np
import json
import pickle
import nltk
from collections import Counter
from tqdm import tqdm


def get_data(mode='train'):

    def parse(sent, only_tokens=False):
        tokens = nltk.word_tokenize(sent)
        if only_tokens:
            return tuple(tokens)
        tags = nltk.pos_tag(tokens)
        chunks = nltk.ne_chunk(tags)
        continuous = nltk.chunk.tree2conlltags(chunks)
        return list(zip(*continuous)) # returns 3 lists (tokenks, tags and NER)

    def find_sub_list(sublist, lst):
        sub_len = len(sublist)
        if sub_len > 0:
            for ind in (i for i,e in enumerate(lst) if e == sublist[0]):
                if lst[ind:ind+sub_len] == sublist:
                    return ind, ind + sub_len

    with open('./data/' + mode + '-v1.1.json') as f:
        data = json.load(f)['data']

    rows = []
    errors = 0
    count = 0

    for i in tqdm(range(len(data))):
        article = data[i]
        for paragraph in article['paragraphs']:

            contexts = [paragraph['context']]

            for context in contexts:

                context = paragraph['context']
                context_tokens, context_tags, context_ner = parse(context)

                for qa in paragraph['qas']:

                    _id, question, answers = qa['id'], qa['question'], qa['answers']
                    question_tokens, question_tags, question_ner = parse(question)

                    if mode == 'train':
                        answer = answers[0]['text']
                        answer_tokens = parse(answer, only_tokens=True)

                        start = answers[0]['answer_start']
                        end = start + len(answer)

                        indexes = find_sub_list(answer_tokens, context_tokens)
                        if indexes:
                            token_start, token_end = indexes
                            assert answer_tokens == context_tokens[token_start:token_end]

                            rows.append((_id, context, question, answer,
                                         context_tokens, context_tags, context_ner,
                                         question_tokens, question_tags, question_ner,
                                         answer_tokens, start, end, token_start, token_end))
                        else:
                            errors += 1
                    else:
                        answers = [a['text'] for a in answers]
                        answer_tokens = [parse(a, only_tokens=True) for a in answers]

                        rows.append((_id, context, question, answers,
                                     context_tokens, context_tags, context_ner,
                                     question_tokens, question_tags, question_ner,
                                     answer_tokens))
            count += 1

    print('Finished cleaning data with', errors, 'errors out of', len(rows))
    return rows


def prepare_data():
    UNK = 'UNK'
    PAD = 'PAD'

    train = get_data('train')
    dev = get_data('dev')

    with open('data/train.pickle', 'wb') as f:
        pickle.dump(train, f)
    with open('data/dev.pickle', 'wb') as f:
        pickle.dump(dev, f)

    all_words = []
    for d in train:
        all_words += set(d[4] + d[7] + d[10])
    for d in dev:
        all_words += set(d[4] + d[7])

    all_characters = Counter(''.join(all_words))
    char2index = { x[0]: i + 2 for i,x in enumerate(all_characters.most_common(100)) }
    char2index[PAD] = 0
    char2index[UNK] = 1

    vocab = set(all_words + [x.lower() for x in all_words])

    words = [PAD, UNK]
    embedding = [np.zeros(300), np.zeros(300)]

    with open(os.path.join('data', 'glove.840B.300d.txt'), 'r', encoding='utf8') as f:
        for line in f:
            word, vec = line.strip().split(' ', 1)
            if word in vocab and word != UNK and word != PAD:
                words.append(word)
                embedding.append(np.array(vec.split(' '), dtype=float))

    word2index = { x: i for i,x in enumerate(words) }
    embedding = np.array(embedding)

    with open('data/word_embed.pickle', 'wb') as f:
        pickle.dump(embedding, f)
    with open('data/char2index.json', 'w') as f:
        json.dump(char2index, f, indent=2)
    with open('data/word2index.json', 'w') as f:
        json.dump(word2index, f, indent=2)

    if not os.path.exists('models'):
        os.makedirs('models')
        print('Created empty dir "models"')

def data():
    with open('data/train.pickle', 'rb') as f:
        train = pickle.load(f)
    with open('data/dev.pickle', 'rb') as f:
        dev = pickle.load(f)
    with open('data/word_embed.pickle', 'rb') as f:
        embed = pickle.load(f)
    with open('data/word2index.json', 'r') as f:
        word2index = json.load(f)
    with open('data/char2index.json', 'r') as f:
        char2index = json.load(f)

    return train, dev, embed, word2index, char2index
