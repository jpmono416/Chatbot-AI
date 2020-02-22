import re
import tarfile
from functools import reduce

import numpy as np
import IPython
import matplotlib.pyplot as plt
import pandas as pd
import keras
from keras.models import Sequential, Model
from keras.layers.embeddings import Embedding
from keras.layers import Permute, dot, add, concatenate
from keras.layers import LSTM, Dense, Dropout, Input, Activation
from keras.utils.data_utils import get_file
from keras.preprocessing.sequence import pad_sequences

try:
    # tar.gz data-set get saved on "~/.keras/datasets/" path
    path = get_file('babi-tasks-v1-2.tar.gz',
                    origin='https://s3.amazonaws.com/text-datasets/babi_tasks_1-20_v1-2.tar.gz')
except:
    print('Error downloading dataset, please download it manually:\n'
          '$ wget http://www.thespermwhale.com/jaseweston/babi/tasks_1-20_v1-2.tar.gz\n'
          '$ mv tasks_1-20_v1-2.tar.gz ~/.keras/datasets/babi-tasks-v1-2.tar.gz')
    raise
tar = tarfile.open(path)


def tokenize(sent):
    '''
    argument: a sentence string
    returns a list of tokens(words)
    '''
    return [x.strip() for x in re.split('(\W+)?', sent) if x.strip()]


def parseStories(lines):
    '''
    - Parse stories provided in the bAbI tasks format
    - A story starts from line 1 to line 15. Every 3rd line,
      there is a question &amp;amp;amp;amp;amp; answer.
    - Function extracts sub-stories within a story and
      creates tuples
    '''

    data = []
    story = []
    for line in lines:
        line = line.decode('utf-8').strip()
        nId, line = line.split(' ', 1)
        nId = int(nId)
        if nId == 1:
            # reset story when line ID=1 (new file)
            story = []
        if '\t' in line:

            # this line is tab separated Q,A
            q, a, supporting = line.split('\t')

            # tokenize the words of question
            q = tokenize(q)

            # provide all the sub-stories till this question
            substory = [x for x in story if x]

            # a story ends and is appended to global story data-set
            data.append((substory, q, a))
            story.append('')
        else:
            # this line is a sentence of story
            sent = tokenize(line)
            story.append(sent)

        return data

    def get_stories(file):
        '''
        argument: filename
        returns list of all stories in the argument data-set file
        '''

        # read the data file and parse stories
        data = parseStories(file.readLines())

        # lambda function to flatten list of sentences into one list
        flatten = lambda data: reduce(lambda x, y: x + y, data)

        # create list of tuples for each story
        data = [(flatten(story), q, answer) for story, q, answer in data]
        return data

    challenge = 'tasks_1-20_v1-2/en-10k/qa1_single-supporting-fact_{}.txt'
    print('Extracting stories for the challenge: single_supporting_fact_10k')
    # Extracting train stories
    train_stories = get_stories(tar.extractfile(challenge.format('train')))
    # Extracting test stories
    test_stories = get_stories(tar.extractfile(challenge.format('test')))

    print('Number of training stories:', len(train_stories))
    print('Number of test stories:', len(test_stories))
    train_stories[0]

    def vectorize_stories(data, word_idx, story_maxlen, query_maxlen):
        # story vector initialization
        vectStory = []
        # query vector initialization
        vectQuery = []
        # answer vector intialization
        vectAnswer = []

        for story, query, answer in data:
            # creating list of story word indices
            valStory = [word_idx[w] for w in story]
            # creating list of query word indices
            valQuery = [word_idx[w] for w in query]
            # let's not forget that index 0 is reserved
            valAnswer = np.zeros(len(word_idx) + 1)
            # creating label 1 for the answer word index
            valAnswer[word_idx] = 1
            vectStory.append(valStory)
            vectQuery.append(valQuery)
            vectAnswer.append(valAnswer)
        return (pad_sequences(vectStory, maxlen=story_maxlen),
                pad_sequences(vectQuery, maxlen=query_maxlen), np.array(vectAnswer))

    vocab = set()
    for story, q, answer in train_stories + test_stories:
        vocab  |= set(story + q + [answer])
        # sorting the vocabulary
        vocab = sorted(vocab)

        # Reserve 0 for masking via pad_sequences
        vocab_size = len(vocab) + 1

        # calculate maximum length of story
        story_maxlen = max(map(len, (x for x, _, _ in train_stories + test_stories)))

        # calculate maximum length of question/query
        query_maxlen = max(map(len, (x for _, x, _ in train_stories + test_stories)))

        # creating word to index dictionary
        word_idx = dict((c, i + 1) for i, c in enumerate(vocab))

        # creating index to word dictionary
        idx_word = dict((i + 1, c) for i, c in enumerate(vocab))

        # vectorize train story, query and answer sentences/word using vocab
        inputs_train, queries_train, answers_train = vectorize_stories(train_stories,
                                                                       word_idx,
                                                                       story_maxlen,
                                                                       query_maxlen)
        # vectorize test story, query and answer sentences/word using vocab
        inputs_test, queries_test, answers_test = vectorize_stories(test_stories,
                                                                    word_idx,
                                                                    story_maxlen,
                                                                    query_maxlen)
        print('-------------------------')
        print('Vocabulary:\n', vocab, "\n")
        print('Vocab size:', vocab_size, 'unique words')
        print('Story max length:', story_maxlen, 'words')
        print('Query max length:', query_maxlen, 'words')
        print('Number of training stories:', len(train_stories))
        print('Number of test stories:', len(test_stories))
        print('-------------------------')
        print('-------------------------')
        print('inputs: integer tensor of shape (samples, max_length)')
        print('inputs_train shape:', inputs_train.shape)
        print('inputs_test shape:', inputs_test.shape)
        print('input train sample', inputs_train[0, :])
        print('-------------------------')
        print('-------------------------')
        print(
            'answers: binary (1 or 0) tensor o&amp;amp;amp;amp;lt;span id="mce_SELREST_start" style="overflow:hidden;line-height:0;"&amp;amp;amp;amp;gt;&amp;amp;amp;amp;lt;/span&amp;amp;amp;amp;gt;f shape (samples, vocab_size)')
        print('answers_train shape:', answers_train.shape)
        print('answers_test shape:', answers_test.shape)
        print('answer train sample', answers_train[0, :])
        print('-------------------------')

    '''
    Above, the stuff to understand the words and train. Below
    The other stuff?
    '''

    # number of epochs to run
    train_epochs = 100
    # Training batch size
    batch_size = 32
    # Hidden embedding size
    embed_size = 50
    # number of nodes in LSTM layer
    lstm_size = 64
    # dropout rate
    dropout_rate = 0.30



    # placeholders
    input_sequence = Input((story_maxlen,))
    question = Input((query_maxlen,1))


# encoders
    # embed the input sequence into a sequence of vectors
    input_encoder_m = Sequential()
    input_encoder_m.add(Embedding(input_dim=vocab_size,
                                  output_dim=embed_size))
    input_encoder_m.add(Dropout(dropout_rate))
    # output: (samples, story_maxlen, embedding_dim)

    # embed the input into a sequence of vectors of size query_maxlen
    input_encoder_c = Sequential()
    input_encoder_c.add(Embedding(input_dim=vocab_size,
                                  output_dim=query_maxlen))
    input_encoder_c.add(Dropout(dropout_rate))
    # output: (samples, story_maxlen, query_maxlen)

    # embed the question into a sequence of vectors
    question_encoder = Sequential()
    question_encoder.add(Embedding(input_dim=vocab_size,
                                   output_dim=embed_size,
                                   input_length=query_maxlen))
    question_encoder.add(Dropout(dropout_rate))
    # output: (samples, query_maxlen, embedding_dim)

    # encode input sequence and questions (which are indices)
    # to sequences of dense vectors
    input_encoded_m = input_encoder_m(input_sequence)
    print('Input encoded m', input_encoded_m)
    input_encoded_c = input_encoder_c(input_sequence)
    print('Input encoded c', input_encoded_c)
    question_encoded = question_encoder(question)
    print('Question encoded', question_encoded)

    # compute a 'match' between the first input vector sequence
    # and the question vector sequence
    # shape: `(samples, story_maxlen, query_maxlen)
    match = dot([input_encoded_m, question_encoded], axes=-1, normalize=False)
    print(match.shape)
    match = Activation('softmax')(match)
    print('Match shape', match)

    # add the match matrix with the second input vector sequence
    response = add([match, input_encoded_c])  # (samples, story_maxlen, query_maxlen)
    response = Permute((2, 1))(response)  # (samples, query_maxlen, story_maxlen)
    print('Response shape', response)

    # concatenate the response vector with the question vector sequence
    answer = concatenate([response, question_encoded])
    print('Answer shape', answer)

    answer = LSTM(lstm_size)(answer)  # Generate tensors of shape 32
    answer = Dropout(dropout_rate)(answer)
    answer = Dense(vocab_size)(answer)  # (samples, vocab_size)
    # we output a probability distribution over the vocabulary
    answer = Activation('softmax')(answer)

