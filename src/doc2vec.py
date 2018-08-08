import gensim
import pandas as pd
import os
import logging
import sys

from src.constants import *
from src import helper

logger = logging.getLogger(__name__)

'''
dm ({1,0}, optional) – Defines the training algorithm.
    If dm=1, ‘distributed memory’ (PV-DM) is used.
    Otherwise, distributed bag of words (PV-DBOW) is employed.
size (int, optional) – Dimensionality of the feature vectors.
window (int, optional) – The maximum distance between the current and predicted
    word within a sentence.
min_count (int, optional) – Ignores all words with total frequency lower than
    this.
dbow_words ({1,0}, optional) –
    If set to 1 trains word-vectors (in skip-gram fashion) simultaneous with
    DBOW doc-vector training; If 0, only trains doc-vectors (faster).
dm_concat ({1,0}, optional) –
    If 1, use concatenation of context vectors rather than sum/average;
    Note concatenation results in a much-larger model, as the input is no
    longer the size of one (sampled or arithmetically combined) word vector,
    but the size of the tag(s) and all words in the context strung together.
'''
PVDM_PARAMS = {
    'dm': 1,
    'vector_size': 400,
    'window': 8,
    'min_count': 5,
    'dm_concat': 1,
    'dbow_words': 0,
    'worker': 8,
    'epochs': 15,
    'alpha': 0.025,
    'sample': 1e-4,
    'hs': 1,
    'negative': 5,
}
PVDBOW_PARAMS = {
    'dm': 0,
    'vector_size': 400,
    'window': 8,
    'min_count': 5,
    'dm_concat': 1,
    'dbow_words': 0,
    'worker': 8,
    'epochs': 15,
    'alpha': 0.025,
    'sample': 1e-4,
    'hs': 1,
    'negative': 5,
}


def iterdocuments(filenames, tokens_only=False):
    """
    Iterate over documents, yielding a list of utf8 tokens at a time.

    Args:
        filenames (TYPE): Description
        encoding (str, optional): Description

    Yields:
        TYPE: Description
    """
    index = 0
    for filename in filenames:
        df = pd.read_csv(filename, names=['text'])
        logger.info('corpus: %s, size: %s',
                    os.path.basename(filename),
                    df.shape[0])
        for _, row in df.iterrows():
            if tokens_only:
                yield gensim.utils.simple_preprocess(row['text'])
            else:
                yield gensim.models.doc2vec.TaggedDocument(
                    gensim.utils.simple_preprocess(row['text']),
                    [index])
                index += 1


def train_model(use_extend=False, is_final=False):
    """Summary

    Args:
        use_external_data (bool, optional): Description
        is_final (bool, optional): Description
    """
    # load corpus to list of sentence
    train_path = ''
    if use_extend:
        train_path = DATA_DIR + 'sentences_data_extend.csv'
    else:
        train_path = DATA_DIR + 'sentences_data.csv'

    train_corpus = list(iterdocuments([train_path]))
    logger.info('train document embedding on corpus: %s', train_path)

    # train model
    models = {
        'dm_concat': PVDM_PARAMS.copy(),
        'dbow_concat': PVDBOW_PARAMS.copy(),
        'dm_sum': PVDM_PARAMS.copy(),
        'dbow_sum': PVDBOW_PARAMS.copy(),
        'dm_mean': None,
        'dbow_mean': None
    }
    models['dm_sum']['dm_concat'] = 0
    models['dm_sum']['dm_mean'] = 0
    models['dbow_sum']['dm_concat'] = 0
    models['dbow_sum']['dm_mean'] = 0

    models['dm_mean'] = models['dm_sum'].copy()
    models['dm_mean']['dm_mean'] = 1
    models['dbow_mean'] = models['dbow_sum'].copy()
    models['dbow_mean']['dm_mean'] = 1
    logger.info('MODELS: %s', models)

    for type_model, params in models.items():
        logger.info('training %s with params: %s',
                    str.upper(type_model), params)
        model = gensim.models.doc2vec.Doc2Vec(**params)
        model.build_vocab(train_corpus)
        model.train(train_corpus, total_examples=model.corpus_count,
                    epochs=model.epochs)

        # save model
        output_path = helper.get_doc2vec_model_path(
            type_model, use_extend, is_final)
        model.save(output_path)
        logger.info('saved PVDM model as {:s}'.format(output_path))

    # access_model()


def access_model():
    """Summary
    """
    corpus_paths = [helper.get_processed_train_path(DATA_DIR + f, is_final)
                    for f in CORPUS_FILES]
    # load corpus to list of sentence
    train_corpus = list(iterdocuments(corpus_paths))

    for type_model in ['dm', 'dbow']:
        # Assessing Model
        model_path = helper.get_doc2vec_model_path(type_model, is_final)
        model = gensim.models.doc2vec.Doc2Vec.load(model_path)
        ranks = []
        second_ranks = []
        for doc_id in range(len(train_corpus)):
            inferred_vector = model.infer_vector(train_corpus[doc_id].words)
            sims = model.docvecs.most_similar(
                [inferred_vector], topn=len(model.docvecs))
            rank = [docid for docid, sim in sims].index(doc_id)
            ranks.append(rank)

            second_ranks.append(sims[1])
            sys.stdout.write('\r')
            sys.stdout.write(str(doc_id))
            sys.stdout.flush()

        sys.stdout.write('\r')
        logger.info(collections.Counter(ranks))
