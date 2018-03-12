import pandas as pd
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


class KaggleJigsawWordEmbedding(object):
    """
    KaggleJigsawWordEmbedding
    """
    def __init__(self, word_embedding_path="../data/input/word_embeddings/glove.6B.100d.txt", docs=None):
        """
        :param word_embedding_path:
        :param docs:
        """
        # prepare tokenizer
        self.token = Tokenizer()
        self.token.fit_on_texts(docs)
        self.vocab_size = len(self.token.word_index) + 1
        # load the whole embedding into memory
        embeddings_index = dict()
        f = open(word_embedding_path)
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
        f.close()
        print('Loaded %s word vectors.' % len(embeddings_index))
        # create a weight matrix for words in training docs
        self.embedding_matrix = np.zeros((self.vocab_size, 100))
        for word, i in self.token.word_index.items():
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                self.embedding_matrix[i] = embedding_vector

    def get_embedding_matrix(self):
        """
        :return: self.embedding_matrix
        """
        return self.embedding_matrix

    def get_encoded_docs(self, docs=None, max_seq_length=100):
        """
        :param docs:
        :param max_length:
        :return: encoded_docs with padding
        """
        # integer encode the documents
        encoded_docs = self.token.texts_to_sequences(docs)
        # print(encoded_docs[:5])
        # pad documents to a max length of 4 words
        padded_docs = pad_sequences(encoded_docs, maxlen=max_seq_length, padding='post')
        # print(padded_docs[:5])
        return padded_docs

    def get_vocabulary_size(self):
        return self.vocab_size
