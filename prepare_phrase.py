
# coding: utf-8

# In[14]:


import pandas as pd
import nltk
from pyvi import ViTokenizer
import ast
import string
import sys
import os
import glob

from src import preprocess
from src.constants import *

# # Extract sentences to train doc2vec
#
# I combine from many data sources
#
# - labeled data
# - comments crawled from ecommerce websites (tiki, lazada, tgdd, vienthonga)
# - features & content of products from from ecommerce websites (...)

# In[19]:

# DATA_DIR = 'data'
# ECOMMERCE_DIR = os.path.join(DATA_DIR, 'ecommerce_sites')
# categories = ['lazada_dien-thoai-di-dong', 'lazada_phu_kien', 'tgdd_cap_sac',
#               'tgdd_chuot', 'tgdd_dtdd', 'tgdd_loa', 'tgdd_pin_sac',
#               'tgdd_tainghe', 'tiki_dien-thoai-may-tinh-bang',
#               'tiki_phu-kien-dien-thoai', 'vta_dien-thoai-smartphones',
#               'vta_phu_kien']
data = []


# combine features, top feaures, content to phrases
redundant_text = "Không hài lòng bài viết. Hãy để lại thông tin để được hỗ trợ khi cần thiết (Không bắt buộc): Anh. Chị. Gửi góp ý. Cam kết bảo mật thông tin cá nhân."
product_paths = glob.glob(ECOMMERCE_DIR + "/*_products.csv")
for product_path in product_paths:
    # product_path = '%s/%s_products.csv' % (ECOMMERCE_DIR, cat)
    feature_df = pd.read_csv(product_path)
    print('load %s products of cat %s' % (feature_df.shape[0],
                                          product_path.split('/')[-1]))
    feature_df = feature_df.drop_duplicates()
    print('after drop duplicates: %s products' % (feature_df.shape[0]))

    for _, row in feature_df.iterrows():
        # extract top features which are seperated by ;
        # normaly, there are : in top features, replace them with whitespace
        if pd.isnull(row['top_features']) is False:
            data.extend([ViTokenizer.tokenize(f.replace(':', ' '))
                         for f in row['top_features'].split(';')])
        # extract content
        if pd.isnull(row['content']) is False:
            content = row['content'].replace(redundant_text, "")
            data.extend([s for s in nltk.sent_tokenize(
                ViTokenizer.tokenize(row['content'])) if len(s) > 1])

        # extract features which are dictionary-formatted string
        if pd.isnull(row['features']) is False:
            features_content = row['features'].replace('\n', ' ')
            primary_features = ['%s %s' % (k, v) for k, v in ast.literal_eval(
                features_content).items() if k != 'SKU']
            data.extend([ViTokenizer.tokenize(f) for f in primary_features])
    print('number of data', len(data))


print('number of data', len(data))


# load comments & split to sentences
comment_paths = glob.glob(ECOMMERCE_DIR + "/*_comments.csv")
for comment_path in comment_paths:
    # comment_path = '%s/%s_comments.csv' % (ECOMMERCE_DIR, cat)
    comment_df = pd.read_csv(comment_path)
    comment_df.head()
    print('load %s comment of cat %s' %
          (comment_df.shape[0], comment_path.split('/')[-1]))

    for _, row in comment_df.iterrows():
        # extract title
        if pd.isnull(row['title']) is False:
            data.append(ViTokenizer.tokenize(row['title']))

        # extract content
        if pd.isnull(row['content']) is False:
            data.extend([s for s in nltk.sent_tokenize(
                ViTokenizer.tokenize(row['content'])) if len(s) > 1])
print('number of data', len(data))

# export to csv
df = pd.DataFrame.from_dict([{'sentences': s} for s in data])
df = df.dropna().drop_duplicates()
print('number of data after drop_duplicates', len(data))
file_path = os.path.join(DATA_DIR, 'sentences_data.csv')
df.to_csv(file_path, index=False)

# load neg, pos, neu => split to sentences
corpus_paths = ['%s/%s_train.csv' %
                (DATA_DIR, l) for l in ['Positive', 'Neutral', 'Negative']]
for filename in corpus_paths:
    df = pd.read_csv(filename, header=None, names=['text'])
    df = df.dropna()
    df = df.drop_duplicates().reset_index(drop=True)
    print('data size after dropna: %s' % df.shape[0])
    print('file: ', filename)
    for _, row in df.iterrows():
        text = row['text'].strip(string.punctuation)
        temp_sentences = [s for s in nltk.sent_tokenize(
            ViTokenizer.tokenize(text)) if len(s) > 1]
        sents = [preprocess.preprocess_sentence(s) for s in temp_sentences]

        if len(sents) > 0:
            data.extend(sents)

        sys.stdout.write('\r')
        sys.stdout.write(str(_))
        sys.stdout.flush()
    sys.stdout.write('\r')
print('number of data', len(data))

# export to csv
df = pd.DataFrame.from_dict([{'sentences': s} for s in data])
df = df.dropna().drop_duplicates()
print('number of data after drop_duplicates', len(data))
file_path = os.path.join(DATA_DIR, 'sentences_data_extend.csv')
df.to_csv(file_path, index=False)
