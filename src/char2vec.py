"""Summary

Attributes:
    logger (TYPE): Description
"""
from fasttext import train_unsupervised
import logging
import pandas as pd
from tqdm import tqdm

from src.constants import *
from src import helper


logger = logging.getLogger(__name__)


def train(min_count, is_final=False):
    """Summary

    Args:
        min_count (TYPE): Description
    """
    logger.info('start train char2vec using fastText package')
    unified_corpus = DATA_DIR + 'unified_corpus.preprocessed.train'
    corpus_paths = [helper.get_processed_train_path(DATA_DIR + f, is_final)
                    for f in CORPUS_FILES]
    frames = []

    tqdm.pandas()
    for corpus_path in corpus_paths:
        df = pd.read_csv(corpus_path, header=None, names=['text'])
        # df.text = df.progress_apply(
        #     lambda x: x['text'].replace('_', ''),
        #     axis=1)
        frames.append(df)
    pd.concat(frames).to_csv(unified_corpus,
                             index=False, header=False)

    w2v_params = FASTTEXT_PARAMS.copy()
    w2v_params['minCount'] = min_count
    logger.info('parameters: %s', w2v_params)

    model = train_unsupervised(input=unified_corpus, **w2v_params)
    model.save_model(helper.get_char2vec_model_path(w2v_params['minCount']))
    logger.info('done train char2vec using fastText package')

    print('get_sentence_vector:', model.get_sentence_vector(
        'tạ_cảnh cảm_ơn thông_tin của bạn ad đã ghi_nhận mong bạn thông_cảm .'))
    print('get_sentence_vector:', model.get_sentence_vector(
        'km nạp thẻ trong hôm_nay áp_dụng cho các số trả trước hòa_mạng trong năm number_token . thuê_bao của bạn kích_hoạt từ number_token bạn ạ .'))
