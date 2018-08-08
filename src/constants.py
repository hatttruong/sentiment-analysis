import string
from enum import Enum
import os


class Replacement(Enum):

    """Define common constants

    Attributes:
        CURRENCY (str): Description
        DATETIME (str): Description
        EMAIL (str): Description
        EMOJI_NEG (str): Description
        EMOJI_POS (str): Description
        NUMBER (str): Description
        PHONE (str): Description
        URL (str): Description

    """
    EMOJI_POS = ' EMO_POS_TOKEN '
    EMOJI_NEG = ' EMO_NEG_TOKEN '
    EMAIL = ' EMAIL_TOKEN '
    URL = ' URL_TOKEN '
    NUMBER = ' NUMBER_TOKEN '
    PHONE = ' PHONE_TOKEN '
    CURRENCY = ' CURRENCY_TOKEN '
    DATETIME = ' DATETIME_TOKEN '


SKIPGRAM_PARAMS = {
    'alpha': 0.025,
    'size': 100,
    'window': 5,
    'iter': 5,
    'min_count': 5,
    'sample': 1e-4,
    'sg': 1,
    'hs': 1,
    'negative': 5
}

FASTTEXT_PARAMS = {
    'model': 'skipgram',
    'lr': 0.1,
    'dim': 100,
    'ws': 5,
    'epoch': 25,
    'minCount': 5,
    'minn': 3,
    'maxn': 8,
    'neg': 5,
    'wordNgrams': 3,
    'loss': 'hs'
}


PUNCTUATIONS = string.punctuation + ' '

DATA_DIR = 'data/'
MODEL_DIR = 'model/'
CORPUS_FILES = [
    'Negative_train.csv',
    'Neutral_train.csv',
    'Positive_train.csv']
EXTERNAL_CORPUS_FILES = [
    'external/train_negative_tokenized.txt',
    'external/train_neutral_tokenized.txt',
    'external/train_positive_tokenized.txt']

# IS_FINAL_MODEL = False

ECOMMERCE_DIR = os.path.join(DATA_DIR, 'ecommerce_sites')
# ECOMMERCE_CATEGORIES = ['lazada_dien-thoai-di-dong', 'lazada_phu_kien',
#                         'tgdd_cap_sac', 'tgdd_chuot', 'tgdd_dtdd', 'tgdd_loa',
#                         'tgdd_pin_sac', 'tgdd_tainghe', 'tiki_dien-thoai-may-tinh-bang',
#                         'tiki_phu-kien-dien-thoai', 'vta_dien-thoai-smartphones',
#                         'vta_phu_kien']
