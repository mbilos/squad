import os
import pickle
import json
import numpy as np
from tqdm import tqdm
from collections import Counter
import nltk

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

    if mode == 'train':
        with open('data/german_to_english.json', encoding='utf-8') as f:
            augmented = json.load(f)

    rows = []
    errors = 0
    count = 0

    for i in tqdm(range(len(data))):
        article = data[i]
        for paragraph in article['paragraphs']:

            contexts = [paragraph['context']]
            if mode == 'train':
                contexts.append(augmented[count])

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

def save_data():
    train = get_data('train')
    dev = get_data('dev')

    with open('./data/train.pickle', 'wb') as f:
        pickle.dump(train, f)
    with open('./data/dev.pickle', 'wb') as f:
        pickle.dump(dev, f)

def word_embedding():

    with open('./data/train.pickle', 'rb') as f:
        train = pickle.load(f)
    with open('./data/dev.pickle', 'rb') as f:
        dev = pickle.load(f)

    assert all([' '.join(train[i][4][train[i][-2]:train[i][-1]]) == ' '.join(train[i][-5]) for i in range(len(train))])
    assert all([train[i][3] in train[i][1] for i in range(len(train))])

    all_words = []
    for d in train:
        all_words += set(d[4] + d[7] + d[10])
    for d in dev:
        all_words += set(d[4] + d[7])

    # CHARACTERS
    all_characters = Counter(''.join(all_words))
    character_to_index = { x[0]: i for i,x in enumerate(all_characters.most_common(100)) }
    with open('data/character_to_index.json', 'w') as f:
        json.dump(character_to_index, f, indent=2)

    all_words = set([x for x in all_words])

    def save_embedding(input_file, output_file):
        embedding = {}
        lower = set([x.lower() for x in all_words])

        with open(input_file, 'r', encoding='utf8') as f:
            for line in f:
                word, vec = line.strip().split(' ', 1)
                if word in all_words or word in lower:
                    embedding.update({ word: np.array(vec.split(' '), dtype=float) })
        with open(output_file, 'wb') as f:
            pickle.dump(embedding, f)
        print('saved', f)

    save_embedding('data/glove.6B.50d.txt',  'data/50d.pickle')
    save_embedding('data/glove.6B.100d.txt', 'data/100d.pickle')
    save_embedding('data/glove.6B.200d.txt', 'data/200d.pickle')
    save_embedding('data/glove.840B.300d.txt', 'data/300d.pickle')

    # PART OF SPEECH
    all_tags = set([x for s in train for x in s[5]])
    tag_to_index = { x: i for i,x in enumerate(all_tags) }
    with open('data/tag_to_index.json', 'w') as f:
        json.dump(tag_to_index, f, indent=2)

    # NAMED ENTITY
    all_entities = set([x for s in train for x in s[6]])
    entity_to_index = { x: i for i,x in enumerate(all_entities) }

    with open('data/entity_to_index.json', 'w') as f:
        json.dump(entity_to_index, f, indent=2)

def read_data(word_embed_size):
    PAD = '='
    UNK = '_'

    with open('data/train.pickle', 'rb') as f:
        train = pickle.load(f)
    with open('data/dev.pickle', 'rb') as f:
        dev = pickle.load(f)
    with open('data/%dd.pickle' % word_embed_size, 'rb') as f:
        embed = pickle.load(f)
        embed[PAD] = np.zeros((word_embed_size))
        embed[UNK] = np.zeros((word_embed_size))

    with open('data/character_to_index.json', 'r') as f:
        character2index = json.load(f)
        index2character = { i: x for x,i in character2index.items() }

    with open('data/tag_to_index.json', 'r') as f:
        tag2index = json.load(f)
        index2tag = { i: x for x,i in tag2index.items() }

    with open('data/entity_to_index.json', 'r') as f:
        entity2index = json.load(f)
        index2entity = { i: x for x,i in entity2index.items() }

    return train, dev, embed, character2index, index2character, \
        tag2index, index2tag, entity2index, index2entity

def data(word_embed_size):
    if not os.path.exists('models'):
        os.makedirs('models')
    if not os.path.exists(os.path.join('data', 'train.pickle')) or not os.path.exists(os.path.join('data', 'dev.pickle')):
        print('Train or dev file doesn\'t exist. Creating now...')
        save_data()
        word_embedding()
        print('Train and dev files saved')
    elif not os.path.exists(os.path.join('data', 'tag_to_index.json')):
        print('Generating embeddings...')
        word_embedding()
    else:
        print('Reading data from file...')

    train, dev, embed, character2index, index2character, tag2index, index2tag, entity2index, index2entity = read_data(word_embed_size)

    return train, dev, embed, character2index, index2character, tag2index, index2tag, entity2index, index2entity