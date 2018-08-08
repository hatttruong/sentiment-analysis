"""Summary
"""
from gensim.models import Word2Vec
import pandas as pd
import os
import logging

from src.constants import *
from src import helper

logger = logging.getLogger(__name__)


def train_model(min_count, use_external_data=False, is_final=False):
    """
    Train word embedding using Skipgram model with gensim

    Args:
        min_count (TYPE): Description
        size (None, optional): Description
        use_external_data (bool, optional): Description
        is_final (bool, optional): Description
    """
    logger.info('train word embedding')

    corpus_paths = [helper.get_processed_train_path(DATA_DIR + f, is_final)
                    for f in CORPUS_FILES]
    output_path = helper.get_w2v_model_path(min_count, is_final)
    if use_external_data:
        output_path = helper.get_external_w2v_model_path()
        corpus_paths.extend(
            [helper.get_processed_train_path(DATA_DIR + f, is_final)
             for f in EXTERNAL_CORPUS_FILES])
    w2v_params = SKIPGRAM_PARAMS.copy()
    w2v_params['min_count'] = min_count

    # load corpus to list of sentence
    documents = []
    for corpus_path in corpus_paths:
        df = pd.read_csv(corpus_path, header=None, names=['text'])
        logger.info('corpus: %s, size: %s',
                    os.path.basename(corpus_path),
                    df.shape[0])
        documents.extend(df.text.tolist())

    logger.info('total of documents: %s', len(documents))
    documents = [d.split() for d in documents]

    # train model
    logger.info('training skipgram on corpus: %s', ','.join(corpus_paths))
    skipgram_model = Word2Vec(documents, **w2v_params)
    skipgram_model.wv.save_word2vec_format(output_path)
    logger.info('saved skipgram model as {:s}'.format(output_path))
