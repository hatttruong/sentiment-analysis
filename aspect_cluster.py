"""
Purpose: cluster features of belong-to-one-category products from many websites
Method: Affinity Progation
    - similarity function: fuzz.token_set_ratio
    - after cross-validation of a manually labeled dataset and the best
    parameters of AP is
        'preference': -11.555555555555555
        'damping': 0.8266666666666667

Output:
    data/features_dien-thoai-may-tinh-bang_affinity_token_set_ratio.csv

Attributes:
    logger (TYPE): Description

"""
from sklearn.cluster import AffinityPropagation
import numpy as np
import pandas as pd
import ast
from fuzzywuzzy import fuzz
from pyvi import ViTokenizer
import glob
import logging
import os
import sys

from src.constants import *

logger = logging.getLogger(__name__)


def load_features():
    """
    load all features from *_product.csv files, tokenized them

    Returns:
        TYPE: Description
    """
    # load all file match *_products.csv
    product_paths = glob.glob(ECOMMERCE_DIR + "/*_products.csv")
    primary_features = list()
    top_features = list()
    for product_path in product_paths:
        df = pd.read_csv(product_path)
        logger.info('file (%s) has %s records',
                    product_path.split('/')[-1], df.shape[0])
        for index, row in df.iterrows():
            # extract top features which are seperated by ;
            # normaly, it is formatted as <feature>:<value>
            if pd.isnull(row['top_features']) is False:
                temp = [f.split(':')[0].strip()
                        for f in row['top_features'].split(';')]
                # keep features whose length is less than 5
                temp = [f for f in temp if len(f) > 1 and len(f.split()) <= 5]
                top_features.extend(temp)

            # extract features which are dictionary-formatted string
            if pd.isnull(row['features']) is False:
                features_content = row['features'].replace('\n', ' ')
                temp_dict_features = ast.literal_eval(features_content)
                primary_features.extend(
                    [k for k in temp_dict_features.keys() if k != 'SKU'])
        logger.info('there are %s top features and %s primary features',
                    len(top_features), len(primary_features))

    total_features = set(primary_features + top_features)
    logger.info('there are %s distinct features in total', len(total_features))

    tokenized_features = [ViTokenizer.tokenize(f) for f in set(total_features)]
    return tokenized_features


def build_similar_matrix(index2feature, similarity_func):
    """Summary

    Args:
        index2feature (TYPE): Description
        similarity_func (TYPE): Description

    Returns:
        TYPE: Description
    """
    num_features = len(index2feature)
    X = np.identity(num_features)
    for i in range(num_features):
        for j in range(num_features):
            if i != j:
                X[i, j] = similarity_func(
                    index2feature[i], index2feature[j]) * 1.0 / 100

        sys.stdout.write('\r')
        sys.stdout.write('%s/%s' % (i, num_features))
        sys.stdout.flush()

    sys.stdout.write('\r')
    return X


def cluster_by_affinitypropagation(documents, index2document=None, X=None,
                                   params=None,
                                   similarity_func=fuzz.token_set_ratio):
    """Summary

    Args:
        documents (TYPE): Description
        index2document (TYPE): Description
        X (None, optional): Description
        params (None, optional): Description
        similarity_func (TYPE, optional): Description
    """
    if index2document is None:
        index2document = dict()
        index = 0
        for d in documents:
            index2document[index] = d
            index += 1

    num_document = len(index2document)
    logger.info('pramameters: num_document=%s, similarity_func=%s',
                num_document, similarity_func.__name__)

    # build X matrix
    if X is None:
        logger.info('build X matrix')
        X = build_similar_matrix(index2document, similarity_func)

    # Compute Affinity Propagation
    logger.info('compute Affinity Propagation')
    af = AffinityPropagation(verbose=2)
    if params is not None:
        af = AffinityPropagation(**params, verbose=2)
    af = af.fit(X)
    labels = af.labels_
    logger.info('number of cluster: %s', len(set(labels)))

    # predict
    documents_by_cluster = dict()
    for i in range(num_document):
        pred_cluster = labels[i]
        if labels[i] in documents_by_cluster.keys():
            documents_by_cluster[pred_cluster].append(index2document[i])
        else:
            documents_by_cluster[pred_cluster] = [index2document[i]]

    # export to file
    logger.info('export to file')
    df = pd.DataFrame.from_dict([{'similar_features': v}
                                 for v in documents_by_cluster.values()])
    category = 'dien-thoai-may-tinh-bang'
    file_path = os.path.join(
        DATA_DIR,
        'features_%s_affinity_%s.csv' % (category, similarity_func.__name__))
    df.to_csv(file_path, index=False)


logging.basicConfig(
    # filename='log/aspect_cluster.log',
    format='%(asctime)s : %(levelname)s : %(message)s',
    level=logging.INFO)

if __name__ == '__main__':
    params = None
    # params = {'preference': -11.555555555555555, 'damping': 0.8266666666666667}
    # params = {'preference': -3, 'damping': 0.8266666666666667}
    cluster_by_affinitypropagation(
        documents=load_features(), params=params,
        similarity_func=fuzz.token_set_ratio)
