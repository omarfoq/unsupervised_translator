import io
import numpy as np


class Word2vec:
    def __init__(self, fname, nmax=100000, verbosity=0):
        """
        :param fname: path to the .vec file containing words embedding
        :param nmax: the number of words to load
        :param verbosity: Set the level of verbosity
        """
        self.verbosity = verbosity
        self.word2vec = {}
        self.load_wordvec(fname, nmax)
        self.word2id = dict.fromkeys(self.word2vec.keys())

        for i, word in enumerate(self.word2id.keys()):
            self.word2id[word] = i
        self.id2word = {v: k for k, v in self.word2id.items()}

        self.embeddings = np.zeros((len(list(self.word2vec.values())), 300))
        for i, word_embed in enumerate(list(self.word2vec.values())):
            self.embeddings[i] = list(self.word2vec.values())[i]

    def load_wordvec(self, fname, nmax):
        """
        load words embedding and store them as a dictionnary in self.word2vec
        :param fname: path to the .vec file containing words embedding
        :param nmax: the number of words to load
        """
        self.word2vec = {}
        with io.open(fname, encoding='utf-8') as f:
            next(f)
            for i, line in enumerate(f):
                word, vec = line.split(' ', 1)
                self.word2vec[word] = np.fromstring(vec, sep=' ')
                if i == (nmax - 1):
                    break
        if self.verbosity == 1:
            print('Loaded %s pretrained word vectors' % (len(self.word2vec)))

