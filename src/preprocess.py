"""Summary

Attributes:
    logger (TYPE): Description
"""
import pandas as pd
import numpy as np
from tqdm import tqdm
from pathlib import Path
import re
import string
from pyvi import ViTokenizer, ViPosTagger
import nltk
import logging
# from sklearn.model_selection import KFold

from src.constants import *
from src import helper

logger = logging.getLogger(__name__)


def handle_emojis(text):
    """Summary

    Args:
        text (TYPE): Description

    Returns:
        TYPE: Description
    """
    # Smile -- :), : ), :-), (:, ( :, (-:, :')
    text = re.sub(r'(:\s?\)|:-\)|\(\s?:|\(-:|:\'\))',
                  Replacement.EMOJI_POS.value, text)
    # Laugh -- :D, : D, :-D, xD, x-D, XD, X-D
    text = re.sub(r'(:\s?D|:-D|x-?D|X-?D)\s',
                  Replacement.EMOJI_POS.value, text)
    # Love -- <3, :*
    text = re.sub(r'(<3|:\*)', Replacement.EMOJI_POS.value, text)
    # Wink -- ;-), ;), ;-D, ;D, (;,  (-;
    text = re.sub(r'(;-?\)|;-?D|\(-?;)', Replacement.EMOJI_POS.value, text)
    # Sad -- :-(, : (, :(, ):, )-:
    text = re.sub(r'(:\s?\(|:-\(|\)\s?:|\)-:)',
                  Replacement.EMOJI_NEG.value, text)
    # Cry -- :,(, :'(, :"(
    text = re.sub(r'(:,\(|:\'\(|:"\()', Replacement.EMOJI_NEG.value, text)

    return text


def handle_url(text):
    """Summary

    Args:
        text (TYPE): Description

    Returns:
        TYPE: Description
    """
    text = re.sub(r'http\S+', Replacement.URL.value, text)
    return text


def handle_email(text):
    """Summary

    Args:
        text (TYPE): Description

    Returns:
        TYPE: Description
    """
    return re.sub(r'(\w+@\w+)', Replacement.EMAIL.value, text)


def handle_numbers(text):
    """Summary

    Args:
        text (TYPE): Description

    Returns:
        TYPE: Description
    """
    # normal numbers
    text = re.sub(r'^\d+\s|\s\d+\s|\s\d+$', Replacement.NUMBER.value, text)
    text = re.sub(r'\b[\d.\/,]+', Replacement.NUMBER.value, text)
    return text


def handle_phone(text):
    """
    Handle cases:
            XX XXX XXX
            XXX XXX XXX
            XXXXXXXXX
        delimiter: whitespace OR - OR empty

    Args:
        text (TYPE): Description

    Returns:
        TYPE: Description
    """
    return re.sub(r'([\+\s]*\d{2,}[-\s.]?\d{3,4}[-\s.]?\d{3,4})',
                  Replacement.PHONE.value, text)


def handle_money(text):
    """
    ATTENTION: remove money before datetime

    Handle cases:
        Group 1
            1,500 (k |tr |triệu|đồng|đ |USD|vnđ)
            23.456.567(k |tr |triệu|đồng|đ |USD|vnđ)
            60.18
        Group 2
            1.000
            12.000.000
        Group 3
            5tr300

    Args:
        text (TYPE): Description

    Returns:
        TYPE: Description
    """
    million_unit = r'triệu|Triệu|TRIỆU|trieu|Trieu|TRIEU'
    billion_unit = r'tỷ|ty|Tỷ|TỶ|tỉ|Tỉ|TỈ'
    usd_unit = r'usd|Usd|USD'
    vnd_unit = r'đồng|Đồng|ĐỒNG|dong|Dong|DONG|vnđ|Vnđ|VNĐ|vnd|Vnd|VND'
    other_unit = r'((k|K|tr|TR|đ|Đ)(\s|\.|-|,|;|:))'
    unit = r'(\s?(' + other_unit + '|' + vnd_unit + '|' + \
        million_unit + '|' + billion_unit + '|' + usd_unit + '))'
    group_1 = r'(\d+([.,]\d{2,})*' + unit + r'\d*)'
    group_2 = r'((\d{1,})([.,]0{3})+)'
    group_3 = r'(\d+\s?' + unit + r'\s?\d+)'
    # logger.debug(r'(' + group_1 + '|' + group_2 + '|' + group_3 + ')')
    return re.sub(r'(' + group_1 + '|' + group_2 + '|' + group_3 + ')',
                  Replacement.CURRENCY.value, text)


def handle_datetime(text):
    """
    Handle cases: MM/YYYY, DD/MM/YYYY, DD/MM
    delimiters: /.-

    Args:
        text (TYPE): Description

    Returns:
        TYPE: Description
    """
    # MM/YYYY
    group_1 = r'(\d{1,2}[-./]\d{4})'

    # DD/MM or DD/MM/YYYY
    group_2 = r'(\d{1,2}[-./]\d{1,2}([-./]\d{4})?)'

    # 09h56 OR 12h
    group_3 = r'(\d{1,2}(h|H|g|G|giờ|Giờ)(\d{1,2}(phút|Phút|ph|PH)?)?)'
    return re.sub(r'(' + group_1 + '|' + group_2 + '|' + group_3 + ')',
                  Replacement.DATETIME.value, text)


def handle_address(text):
    """Summary

    Args:
        text (TYPE): Description
    """
    pass


def remove_non_alphabet(text):
    """
    Reference: http://vietunicode.sourceforge.net/charset/vietalphabet.html

    Args:
        text (TYPE): Description

    Returns:
        TYPE: Description
    """
    text = re.sub(
        r'[^aAàÀảẢãÃáÁạẠăĂằẰẳẲẵẴắẮặẶâÂầẦẩẨẫẪấẤậẬbBcCdDđĐeEèÈẻẺẽẼéÉẹẸêÊềỀểỂễỄếẾệỆfFgGhHiIìÌỉỈĩĨíÍịỊjJkKlLmMnNoOòÒỏỎõÕóÓọỌôÔồỒổỔỗỖốỐộỘơƠờỜởỞỡỠớỚợỢpPqQrRsStTuUùÙủỦũŨúÚụỤưƯừỪửỬữỮứỨựỰvVwWxXyYỳỲỷỶỹỸýÝỵỴzZ0-9_' + string.punctuation + ']',
        ' ', text
    )
    return text


def extract_hashtags(text):
    """Summary

    Args:
        text (TYPE): Description

    Returns:
        TYPE: Description
    """
    hashtags = re.findall(r'#(\S+)', text)
    text = re.sub(r'#(\S+)', ' ', text)
    return text, hashtags


def handle_hashtags(df):
    """Summary

    Args:
        df (TYPE): Description

    Returns:
        TYPE: Description
    """
    hashtags = list()
    for index, row in df.iterrows():
        text, ht = extract_hashtags(row['text'])
        hashtags.extend(ht)
        df.loc[index, 'text'] = text
    return df, hashtags


def find_proper_name_in_fb(text, pattern, direction='both'):
    m = re.search(pattern, text)
    if m is not None:
        start = m.span()[0]
        idx_prev_upper = [start]

        # find previous name
        if direction in ['left', 'both']:
            i = start - 1
            while i >= 0:
                i -= 1
                if text[i] in list(string.punctuation):
                    break
                if text[i] == ' ':
                    if str.islower(text[i + 1]):
                        break
                    if str.isupper(text[i + 1]):
                        idx_prev_upper.append(i + 1)
                        if len(idx_prev_upper) >= 5:
                            break

        # find after name
        end = m.span()[1]
        idx_next_upper = [end]
        if direction in ['right', 'both']:
            i = end
            nb_capital_words = 0
            while i < (len(text) - 1):
                if text[i] in list(string.punctuation):
                    break
                if text[i] == ' ':
                    if str.islower(text[i + 1]):
                        idx_next_upper.append(i)
                        break
                    if str.isupper(text[i + 1]):
                        idx_next_upper.append(i)
                        nb_capital_words += 1
                        if nb_capital_words > 5:
                            break

                i += 1
                if i == (len(text) - 1):
                    # get the ending position
                    idx_next_upper.append(i)

        return idx_prev_upper[-1], idx_next_upper[-1]
    return None, None


def handle_facebook_posting(text):
    """
    extract feelings
    remove ending of texts which come from posting photos on Facebook
    """
    feelings = []

    # feeling
    feeling_patterns = [
        (r"(is\s|was\s)?feeling (\w+)(\s?\.)", "left"),
        (r"(is\s|was\s)?feeling (\w+)(\swith) (.+?) at", "both"),
        (r"(is\s|was\s)?feeling (\w+)((\sat|\swith) \w+)", "both"),
    ]
    for feeling_pattern, direction in feeling_patterns:
        temp_feelings = re.findall(feeling_pattern, str.lower(text))
        if len(temp_feelings) > 0:
            feelings.extend([f[1] for f in temp_feelings])

        # remove feelings
        start, end = find_proper_name_in_fb(
            text, r'(' + feeling_pattern + ')', direction=direction)
        if start is not None and end is not None:
            if (start * 1. / len(text)) >= 0.7:
                text = text[:start]
            else:
                text = text.replace(text[start:end], ' ')
    text = re.sub(r'(and \d+ others)', ' ', text)

    # remove pattern posting photos ending
    text = re.sub(r"(Photos from (.+?)'s post)", ' ', text)

    # remove added photos/video
    photo_pattern = r'(added (\d+|a) new (photos|photo|videos|video))'
    start, end = find_proper_name_in_fb(text, photo_pattern, direction='left')
    if start is not None and end is not None:
        if (start * 1. / len(text)) >= 0.7:
            text = text[:start]
        else:
            text = text.replace(text[start:end], ' ')

    checkin_patterns = [r'(with (.+?) and (.+?)\.)',
                        r'(and \d+ others at)',
                        r'(\d+ others at)']
    for checkin_pattern in checkin_patterns:
        start, end = find_proper_name_in_fb(
            text, checkin_pattern, direction='both')
        if start is not None and end is not None:
            if (start * 1. / len(text)) >= 0.7:
                text = text[:start]
            else:
                text = text.replace(text[start:end], ' ')

    # remove share pattern
    share_pattern = r"(shared (.+?)'s (post|photo|photos))"
    start, end = find_proper_name_in_fb(text, share_pattern, direction='left')
    if start is not None and end is not None:
        if (start * 1. / len(text)) >= 0.7:
            text = text[:start]
        else:
            text = text.replace(text[start:end], ' ')

    # remove some special words
    text = re.sub(r'(Timeline Photos)', ' ', text)

    return text, feelings


def lower_case(text):
    """
    if 70% text is upper case, it might be upper case to take attention
    of reader

    Args:
        text (TYPE): Description

    Returns:
        TYPE: Description
    """
    count_upper = len([w for w in text.split() if w.isupper()])
    if count_upper >= len(text.split()) * 0.7:
        text = str.lower(text)
    return text


def preprocess_sentence(text):
    """
    Args:
        text (TYPE): Description

    Returns:
        TYPE: Description
    """
    funcs = [handle_emojis, handle_url, handle_phone, handle_money,
             handle_datetime, handle_numbers, handle_email,
             remove_non_alphabet, ]
    for f in funcs:
        logger.debug('preprocess %s' % str(f))
        text = f(text)

    return text


def preprocess_document(doc, skip_viTokenizer=False, export_pos=False):
    """Summary

    Args:
        doc (TYPE): Description
        skip_viTokenizer (bool, optional): Description
        export_pos (bool, optional): Description

    Returns:
        TYPE: Description
    """
    # extract hashtags first before stripping punctuations
    text, hashtags = extract_hashtags(doc)

    # extract some special case of facebook information
    text, feelings = handle_facebook_posting(text.strip(PUNCTUATIONS))

    # remove 1-character sentence
    logger.debug('start vietnamese tokenize and sent_tokenize')
    logger.debug('text: %s', text)
    if skip_viTokenizer:
        sents = [s for s in nltk.sent_tokenize(text) if len(s) > 1]
    else:
        sents = [s for s in nltk.sent_tokenize(ViTokenizer.tokenize(text))
                 if len(s) > 1]
    # print('step1: \t', sents)

    # preprocess each sentence
    sents = [preprocess_sentence(s) for s in sents]
    # print('step2: \t', sents)

    # filter again too short sentence
    sents = [s.strip() for s in sents if len(s.strip()) > 1]
    # print('step3: \t', sents)

    # add punctuation at the end of each sentence if it does not have
    logger.debug('start word_tokenize')
    sents = [' '.join([w.strip(PUNCTUATIONS)
                       for w in nltk.word_tokenize(s)]) for s in sents]
    sents = [s if s.endswith(tuple(string.punctuation))
             else s + ' .' for s in sents]
    # print('step4: \t', sents)

    # post tagging. Note that we lower case after doing posttagging in order
    # to make pre step work correctly
    logger.debug('start postagging')
    final_sents = []
    for s in sents:
        tokens, labels = ViPosTagger.postagging(s)
        # remove Np
        tokens = [t.lower() for t, l in zip(tokens, labels) if l not in ['Np']]

        # # not Remove Np
        # tokens = [t.lower() for t, l in zip(tokens, labels)]

        if export_pos:
            final_sents.append(
                ' '.join(['%s/%s' % (t, l)
                          for t, l in zip(tokens, labels)
                          if len(t) > 0]))
        else:
            final_sents.append(
                ' '.join([t for t in tokens if len(t) > 0]))

    if len(final_sents) == 0:
        return None
    return ' '.join(final_sents)


def preprocess(file_names, skip_viTokenizer=False, export_pos=True,
               test_size=0.3, is_final=False):
    """Summary

    Args:
        file_names (TYPE): Description
        skip_viTokenizer (bool, optional): Description
        export_pos (bool, optional): Description
        test_size (float, optional): Description
    """
    logger.info('preprocess: %s', locals())
    helper.prepare_nltk()

    np.random.seed(0)

    tqdm.pandas()
    for fname in file_names:
        path = Path(DATA_DIR + fname)
        df = pd.read_csv(path, header=None, names=['text'])
        logger.info('start preprocess file: %s', fname)
        logger.info('data size: %s', df.shape[0])
        df = df.dropna()
        logger.info('data size after dropna: %s', df.shape[0])
        df = df.drop_duplicates()
        logger.info('data size after drop_duplicates: %s', df.shape[0])

        df.text = df.progress_apply(
            lambda x: preprocess_document(x['text'],
                                          skip_viTokenizer=skip_viTokenizer,
                                          export_pos=export_pos),
            axis=1)
        df = df.dropna()
        logger.info('data size after preprocess: %s', df.shape[0])

        tokenize_path = helper.get_processed_path(DATA_DIR + fname)
        if export_pos:
            tokenize_path += '_pos'

        if is_final or test_size <= 0.:
            df.to_csv(Path(tokenize_path), index=False, header=False)

        elif test_size > 0.:
            # split into train / test set
            df = df.sample(frac=1).reset_index(drop=True)
            msk = np.random.rand(len(df)) < test_size
            df[msk].to_csv(Path(tokenize_path + '.test'),
                           index=False, header=False)
            df[~msk].to_csv(Path(tokenize_path + '.train'),
                            index=False, header=False)

        logger.info('done preprocess file: %s', fname)
