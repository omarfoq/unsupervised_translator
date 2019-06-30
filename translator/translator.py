import os
import numpy as np
import scipy
import scipy.linalg
from translator.word2vec import Word2vec


class Translator:
    def __init__(self, src_lang, target_lang, src_lang_size=50000, target_lang_size=50000, verbosity=0):
        """
        Translator Object to translate words from src_lang to target_lang
        :param src_lang: source language of our translator object. Possible : 'en' (english) or 'fr' (french)
        :param target_lang: target language of our translator object. Possible : 'en' (english) or 'fr' (french)
        :param src_lang_size: Maximum size of source language vocabulary to load
        :param target_lang_size: Maximum size of target language vocabulary to load
        :param verbosity: control verbosity. set 0 for no verbosity
        """
        PATH_TO_DATA = 'data/'
        self.src_lang = src_lang
        self.target_lang = target_lang
        self.verbosity = verbosity

        if src_lang == 'en':
            self.src_word2vec = Word2vec(os.path.join(PATH_TO_DATA, 'wiki.en.vec'), nmax=src_lang_size)
        elif src_lang == 'fr':
            self.src_word2vec = Word2vec(os.path.join(PATH_TO_DATA, 'wiki.fr.vec'), nmax=src_lang_size)
        else:
            raise NotImplementedError

        if target_lang == 'en':
            self.target_word2vec = Word2vec(os.path.join(PATH_TO_DATA, 'wiki.en.vec'), nmax=target_lang_size)
            self.target_language = 'English'
        elif target_lang == 'fr':
            self.target_word2vec = Word2vec(os.path.join(PATH_TO_DATA, 'wiki.fr.vec'), nmax=target_lang_size)
            self.target_language = 'French'
        else:
            raise NotImplementedError

        self.fit()

    def fit(self):
        """
        fits a linear transformation between the source language manifold and the target language manifold
        """
        X, Y = [], []

        count = 0
        id_char = []
        for k, v in self.target_word2vec.word2vec.items():
            if k in self.src_word2vec.word2vec:
                count += 1
                Y.append(v)
                X.append(self.src_word2vec.word2vec[k])
                id_char.append(k)

        if self.verbosity == 1:
            print("Number of identical character strings : ", count)

        X = np.vstack(X).T
        Y = np.vstack(Y).T

        U, s, V = scipy.linalg.svd(np.dot(Y, X.T))
        W = np.dot(U, V)
        self.W = W

    def nearest_neighbours_translation(self, word, k=1):
        """
        Gets the k-nearest neighbours words to the input "word", in the target language
        :param word: (str) word to traduce
        :param k: set it to 1 for translation
        :return: list of k-nearest neighbours words to the input in the target language
        """
        query = self.src_word2vec.word2vec[word]
        word_emb = np.dot(self.W, query)
        indexes = np.argsort(np.sum(np.abs(self.target_word2vec.embeddings - word_emb), axis=1))[:k]
        nn_word = [list(self.target_word2vec.word2vec.keys())[i] for i in indexes]

        if len(nn_word) == 1:
            print("Nearest {} neighbours to {} is {}".format(self.target_language, word, nn_word[0]))
        else:
            print("Nearest {} neighbours to {} are {}".format(self.target_language, word, nn_word))

        return nn_word
