#!/usr/bin/python
"""
Filter documents by features
Use multithreading

"""
import threading
import pandas as pd
import logging
from pyvi import ViTokenizer
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from fuzzywuzzy import fuzz

exitFlag = 0
logger = logging.getLogger(__name__)


class extractDocumentsThread(threading.Thread):

    def __init__(self, threadId, name, group_feature, corpus_paths):
        threading.Thread.__init__(self)
        self.threadId = threadId
        self.name = name
        self.group_feature = group_feature
        self.corpus_paths = corpus_paths

    def run(self):
        logger.info("Starting " + self.name)
        extractDocumentsByFeature(
            self.name, self.group_feature, self.corpus_paths)
        logger.info("Exiting " + self.name)


def iterdocuments(filenames):
    """
    Iterate over documents
    """
    for filename in filenames:
        df = pd.read_csv(filename, header=None, names=['text'])
        for _, row in df.iterrows():
            yield row['text']


def extractDocumentsByFeature(threadName, group_feature, corpus_paths):
    # tokenized_features = [ViTokenizer.tokenize(f) for f in group_feature]
    documents = list(iterdocuments(corpus_paths))
    logger.info('%s, done load documents', threadName)

    # set max ngrams
    max_ngram = sorted([len(f.split())
                        for f in group_feature], reverse=True)[0]
    max_ngram = 3 if max_ngram < 3 else max_ngram
    logger.info('%s, max_ngram=%s', threadName, max_ngram)
    vectorizer = CountVectorizer(ngram_range=(2, max_ngram))
    analyze = vectorizer.build_analyzer()

    # find matching document
    filter_documents = []
    for index, d in enumerate(documents):
        terms = analyze(d)
        for term in terms:
            score = [0.]
            for feature in group_feature:
                score.append(fuzz.token_set_ratio(term, feature))
            if np.average(score) >= 60:
                filter_documents.append(d)
                break
        if index % 100 == 0:
            logger.info(
                '%s, done on %s documents, filter documents: %s',
                threadName, index, len(filter_documents))

    # export to file
    df = pd.DataFrame.from_dict([{'text': v} for v in filter_documents])
    group_name = '_'.join(group_feature[0].split())
    file_path = 'data/documents_related_%s.csv' % (group_name)
    df.to_csv(file_path, index=False)
    logger.info('%s, export %s documents of group %s to file',
                threadName, len(filter_documents), group_name)


logging.basicConfig(
    filename='log/log_extract_documents.log',
    format='%(asctime)s : %(levelname)s : %(message)s',
    level=logging.DEBUG)

# load group features
file_path = 'data/dien-thoai-may-tinh-bang_affinity_token_set_ratio.csv'
df = pd.read_csv(file_path)
group_features = list()
for index, row in df.iterrows():
    group_features.append(set([x.strip("'. ") for x in row["similar_features"][
                          1:-1].split(',') if len(x.strip("'. ")) > 0]))

# Create new threads
corpus_paths = ['data/%s_train.csv' % l
                for l in ['Positive', 'Neutral', 'Negative']]
threads = list()
for i, group_feature in enumerate(group_features):
    thread = extractDocumentsThread(
        i, "Thread-%s" % i, group_feature, corpus_paths)
    threads.append(thread)

# Start new Threads
for th in threads:
    th.start()

print("Exiting Main Thread")
