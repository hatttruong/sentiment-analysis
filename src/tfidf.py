"""Summary

Attributes:
    logger (TYPE): Description
"""
import pandas as pd
import logging
import nltk
import gensim
import numpy as np

from src.constants import *
from src import helper

logger = logging.getLogger(__name__)


def to_vector(data, length):
    """
    Intial sparse matrix from list of tuple(col_index, value)

    Example:
            data = [(0, 3), (1, 2), (5, 7), (20, 1)]
            return: <1x21 sparse matrix of type '<class 'numpy.int64'>'
                    with 4 stored elements in Compressed Sparse Row format>
        Use function toarray() to convert sparse matrix to dense matrix

    Args:
        data (list): list of (column index, value)
        length (TYPE): Description

    Returns:
        sparse matrix
    """
    col = [c for c, v in data]
    values = [v for c, v in data]
    v = [0 for i in range(length + 1)]
    for c, value in zip(col, values):
        v[c] = value
    return np.array(v)


def extract_ngrams(tokens, ngrams=(1, 1)):
    """Turn a sequence of tokens into space-separated n-grams.

    Args:
        tokens (TYPE): list of one-word
        ngrams (tuple, optional): range of ngrams

    Returns:
        TYPE: Description

    Raises:
        e: Description
        ValueError: Description
    """
    if ngrams[1] < ngrams[0]:
        raise ValueError(
            'ngrams[1] must be greater than or equal to ngrams[0]')

    result = []
    for i in range(ngrams[0], ngrams[1] + 1):
        if i == 1:
            result.extend(tokens)
        else:
            try:
                result.extend([' '.join(str(a))
                               for a in nltk.ngrams(tokens, i)])
            except TypeError as e:
                logger.error('i=%s', i)
                raise e
    logger.debug('number of ngrams: %s', result)
    return result


def iterdocuments(filenames, encoding='utf-8'):
    """
    Iterate over documents, yielding a list of utf8 tokens at a time.

    Args:
        filenames (TYPE): Description
        encoding (str, optional): Description

    Yields:
        TYPE: Description
    """

    for filename in filenames:
        df = pd.read_csv(filename, header=None, names=['text'])
        for index, row in df.iterrows():
            yield row['text'].split()


class ChunkedCorpus(object):
    """Split text files into chunks and extract n-gram BOW model.

    Attributes:
        chunksize (TYPE): Description
        dictionary (TYPE): Description
        filenames (TYPE): Description
        min_count (TYPE): Description
        ngrams (TYPE): Description
    """

    def __init__(self, filenames, min_count=5, chunksize=5000, ngrams=(1, 1),
                 dictionary=None):
        """Summary

        Args:
            filenames (TYPE): Description
            min_count (int, optional): Description
            chunksize (int, optional): Description
            ngrams (tuple, optional): Description
            dictionary (None, optional): Description
        """
        logger.debug('parameters: %s', locals())
        self.filenames = filenames
        self.min_count = min_count
        self.ngrams = ngrams
        self.chunksize = chunksize
        if dictionary is None:
            # no_below = self.count_below_threshold()
            no_below = self.min_count
            logger.info('no_below of corpus is: %s', no_below)
            self.dictionary = gensim.corpora.Dictionary(
                extract_ngrams(tokens, self.ngrams)
                for tokens in iterdocuments(filenames) if len(tokens) > 0)
            logger.info('number of tokens before filtering: %s',
                        len(self.dictionary.token2id.keys()))
            self.dictionary.filter_extremes(no_below=no_below, keep_n=200000)
            self.dictionary.compactify()
            logger.info('number of tokens after filtering: %s',
                        len(self.dictionary.token2id.keys()))
        else:
            self.dictionary = dictionary

    def count_below_threshold(self):
        """Summary

        Returns:
            TYPE: Description
        """
        no_below = 1
        if self.min_count > 0 and len(self.filenames) > 0:
            total = 0
            for f in self.filenames:
                total += pd.read_csv(f, header=None, names=['text']).shape[0]
            no_below = int(self.min_count * total)

        return no_below

    def __iter__(self):
        """Summary

        Yields:
            TYPE: Description
        """
        for tokens in iterdocuments(self.filenames):
            for n, chunk in enumerate(gensim.utils.chunkize(
                    extract_ngrams(tokens, self.ngrams),
                    self.chunksize,
                    maxsize=2)):
                yield self.dictionary.doc2bow(chunk)


class TfidfGensimVectorizer(object):

    """Summary

    Attributes:
        dictionary (TYPE): Description
        id2word (TYPE): Description
        idfs (TYPE): Description
        ngrams (TYPE): Description
        tfidf_model (TYPE): Description
        token2id (TYPE): Description

    """

    def __init__(self, dictionary_file, ngrams=(1, 1), tfidf_model_path=None):
        """Summary

        Args:
            dictionary_file (TYPE): Description
            ngrams (TYPE): Description
            tfidf_model_path (None, optional): Description
        """
        self.ngrams = ngrams
        self.dictionary = gensim.corpora.Dictionary.load(dictionary_file)
        logger.debug('Number of token2id.keys: %s',
                     len(self.dictionary.token2id.keys()))
        logger.debug('Number of id2token.keys: %s',
                     len(self.dictionary.id2token.keys()))
        logger.debug('Number of dfs.keys: %s',
                     len(self.dictionary.dfs.keys()))
        self.id2word = gensim.utils.revdict(self.dictionary.token2id)
        self.token2id = self.dictionary.token2id
        logger.debug('Number of self.id2word.keys: %s',
                     len(self.id2word.keys()))
        logger.debug('Number of self.token2id.keys: %s',
                     len(self.token2id.keys()))

        self.tfidf_model = None
        self.idfs = list()
        if tfidf_model_path is not None:
            self.load(tfidf_model_path)

    def load(self, path):
        """Summary

        Args:
            path (TYPE): Description
        """
        self.tfidf_model = gensim.models.TfidfModel.load(path)
        self.idfs = self.tfidf_model.idfs
        logger.debug('Number of self.idfs.keys: %s',
                     len(self.idfs.keys()))

    def save(self, path):
        """Summary

        Args:
            path (TYPE): Description
        """
        if self.tfidf_model is not None:
            self.tfidf_model.save(path)

    def fit(self):
        """
        Fit model by using its dictionary which contains information about
        corpus
        """
        self.tfidf_model = gensim.models.TfidfModel(
            dictionary=self.dictionary, normalize=True)
        self.idfs = self.tfidf_model.idfs
        logger.debug('Number of self.idfs.keys: %s',
                     len(self.idfs.keys()))

    def transform(self, document):
        """
        Clean the input document, then extract ngrams of it and convert to BOW
        Finally, convert to tf-idf vector

        Args:
            document (TYPE): the raw document

        Returns:
            TYPE: vector tf-idf
        """
        token2tfidf = dict()
        if document is not None and len(document) > 0:
            tokens = document.split()
            doc2bow = self.dictionary.doc2bow(
                extract_ngrams(tokens, self.ngrams))

            for idx, tfidf in self.tfidf_model[doc2bow]:
                token2tfidf[self.id2word[idx]] = tfidf
        return token2tfidf


def train_tfidf(min_count=1, chunksize=5000, ngrams=(1, 1), is_final=False):
    """Summary

    Args:
        min_count (int, optional): Description
        chunksize (int, optional): Description
        ngrams (tuple, optional): Description
    """
    train_files = [helper.get_processed_train_path(DATA_DIR + f, is_final)
                   for f in CORPUS_FILES]
    dictionary_path = helper.get_dictionary_path(min_count, is_final)
    tfidf_model_path = helper.get_tfidf_model_path(min_count, is_final)
    chunkedCorpus = ChunkedCorpus(
        train_files, min_count=min_count, chunksize=ngrams, ngrams=ngrams)
    chunkedCorpus.dictionary.save(dictionary_path)

    # build tf-idf model
    logger.info('build tf-idf model')
    tfidf = TfidfGensimVectorizer(
        dictionary_file=dictionary_path, ngrams=ngrams)
    tfidf.fit()
    tfidf.save(tfidf_model_path)
