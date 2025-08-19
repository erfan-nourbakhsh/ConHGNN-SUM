import numpy as np
from tools.logger import *


class Word_Embedding(object):
    """
    Class to handle external word embeddings and map them to a given vocabulary.
    Provides methods to load embeddings and handle unknown words.
    """

    def __init__(self, path, vocab):
        """
        Initialize the Word_Embedding object.

        :param path: str, file path of the pre-trained word embedding
        :param vocab: Vocabulary object that provides word_list() and id2word()
        """
        logger.info("[INFO] Loading external word embedding...")
        self._path = path  # store the path to embedding file
        self._vocablist = vocab.word_list()  # list of words in the vocabulary
        self._vocab = vocab  # reference to the vocabulary object

    def load_my_vecs(self, k=200):
        """
        Load word embeddings from the external file, limited to top-k dimensions.

        :param k: int, number of dimensions to load per word
        :return: dict, mapping word -> vector (list of floats)
        """
        word_vecs = {}  # dictionary to store word embeddings
        with open(self._path, encoding="utf-8") as f:
            count = 0
            lines = f.readlines()[1:]  # skip header line if present
            for line in lines:
                values = line.split(" ")
                word = values[0]  # first element is the word itself
                count += 1
                if word in self._vocablist:  # only load words present in vocab
                    vector = []
                    for count, val in enumerate(values):
                        if count == 0:  # skip word itself
                            continue
                        if count <= k:  # limit to k dimensions
                            vector.append(float(val))
                    word_vecs[word] = vector
        return word_vecs

    def add_unknown_words_by_zero(self, word_vecs, k=200):
        """
        Assign zero vectors to unknown words not in the external embedding.

        :param word_vecs: dict of loaded embeddings
        :param k: int, embedding dimension
        :return: list of word vectors corresponding to vocab order
        """
        zero = [0.0] * k  # zero vector
        list_word2vec = []  # list to store final embeddings
        oov = 0  # out-of-vocabulary counter
        iov = 0  # in-vocabulary counter

        # iterate through all words in vocab
        for i in range(self._vocab.size()):
            word = self._vocab.id2word(i)
            if word not in word_vecs:
                oov += 1
                word_vecs[word] = zero
                list_word2vec.append(word_vecs[word])
            else:
                iov += 1
                list_word2vec.append(word_vecs[word])

        logger.info("[INFO] oov count %d, iov count %d", oov, iov)
        return list_word2vec

    def add_unknown_words_by_avg(self, word_vecs, k=200):
        """
        Assign unknown words the average vector of known embeddings.

        :param word_vecs: dict of loaded embeddings
        :param k: int, embedding dimension
        :return: list of word vectors in vocab order
        """
        # collect existing embeddings into a list
        word_vecs_numpy = [word_vecs[word] for word in self._vocablist if word in word_vecs]

        # compute average per dimension
        col = []
        for i in range(k):
            sum_val = 0.0
            for j in range(len(word_vecs_numpy)):
                sum_val += word_vecs_numpy[j][i]
                sum_val = round(sum_val, 6)
            col.append(sum_val)

        # create average vector for unknown words
        zero = []
        for m in range(k):
            avg = col[m] / len(word_vecs_numpy)
            avg = round(avg, 6)
            zero.append(float(avg))

        # assign embeddings to all vocab words
        list_word2vec = []
        oov = 0
        iov = 0
        for i in range(self._vocab.size()):
            word = self._vocab.id2word(i)
            if word not in word_vecs:
                oov += 1
                word_vecs[word] = zero
                list_word2vec.append(word_vecs[word])
            else:
                iov += 1
                list_word2vec.append(word_vecs[word])

        logger.info("[INFO] External Word Embedding iov count: %d, oov count: %d", iov, oov)
        return list_word2vec

    def add_unknown_words_by_uniform(self, word_vecs, uniform=0.25, k=200):
        """
        Assign unknown words a random vector from uniform(-uniform, uniform).

        :param word_vecs: dict of loaded embeddings
        :param uniform: float, range of uniform distribution
        :param k: int, embedding dimension
        :return: list of word vectors in vocab order
        """
        list_word2vec = []
        oov = 0
        iov = 0

        for i in range(self._vocab.size()):
            word = self._vocab.id2word(i)
            if word not in word_vecs:
                oov += 1
                word_vecs[word] = np.random.uniform(-uniform, uniform, k).round(6).tolist()
                list_word2vec.append(word_vecs[word])
            else:
                iov += 1
                list_word2vec.append(word_vecs[word])

        logger.info("[INFO] oov count %d, iov count %d", oov, iov)
        return list_word2vec

    def load_my_vecs_freq1(self, freqs, pro):
        """
        Load embeddings, but for words appearing only once, include probabilistic filtering.

        :param freqs: dict, word -> frequency in the corpus
        :param pro: float, probability threshold for keeping single-occurrence words
        :return: dict, loaded embeddings
        """
        word_vecs = {}
        with open(self._path, encoding="utf-8") as f:
            lines = f.readlines()[1:]  # skip header
            for line in lines:
                values = line.split(" ")
                word = values[0]
                if word in self._vocablist:
                    # For words that occur only once, randomly skip based on probability
                    if freqs[word] == 1:
                        a = np.random.uniform(0, 1, 1).round(2)
                        if pro < a:
                            continue

                    vector = [float(val) for count, val in enumerate(values) if count != 0]
                    word_vecs[word] = vector

        return word_vecs
