"""Summary

Attributes:
    logger (TYPE): Description
"""
import sys
import logging
import os
import numpy as np
# import fastText
from gensim.models import KeyedVectors
from sklearn.mixture import GaussianMixture
from sklearn.linear_model import LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
from sklearn.externals import joblib
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from tqdm import tqdm

from src.constants import *
from src import helper
from src.tfidf import *
from src import w2v
from src import tfidf
from src import preprocess

logger = logging.getLogger(__name__)


def get_noun_pharses():
    """Summary

    Returns:
        TYPE: Description
    """
    return []

    logger.info('get list noun phrases from train files')
    noun_phrases = list()
    corpus_paths = [helper.get_processed_path(DATA_DIR + f) + '_pos.train'
                    for f in CORPUS_FILES]
    for corpus_path in corpus_paths:
        df = pd.read_csv(corpus_path, header=None, names=['text'])
        for index, row in df.iterrows():
            terms = row['text'].split()

            noun_phrases.extend([t.split('/')[0]
                                 for t in terms if t.split('/')[1] == 'Np'])
    noun_phrases = list(set(noun_phrases))
    logger.info('Number of noun phrases from train files: %s',
                len(noun_phrases))

    return noun_phrases


def build_w2v_matrix(wv_model, common_words):
    """Summary

    Args:
        wv_model (TYPE): Description
        common_words (TYPE): Description

    Returns:
        TYPE: Description
    """
    return np.transpose(np.array([wv_model[w] for w in common_words]))


def transform_doc2vec_using_w2v_tfidf(token2tfidf, w2v_matrix, common_words):
    """Summary

    Args:
        token2tfidf (TYPE): Description
        w2v_matrix (TYPE): Description
        common_words (TYPE): Description

    Returns:
        TYPE: Description
    """
    tfidf_vector = np.array(
        [token2tfidf[w] if w in token2tfidf.keys() else 0.
         for w in common_words])
    logger.debug('tfidf_vector.shape: %s', tfidf_vector.shape)
    v = np.dot(w2v_matrix, tfidf_vector)
    norm = np.linalg.norm(v)
    if norm == 0:
        logger.info(
            'norm of doc2vec is zero: number of token is %s,tfidf_vector.shape=%s',
            len(token2tfidf.keys()), tfidf_vector.shape)
    return v / norm if norm != 0 else v


def transform_doc2vec_using_w2v(token2tfidf, wv_model):
    """Summary

    Args:
        token2tfidf (TYPE): Description
        wv_model (TYPE): Description

    Returns:
        TYPE: Description

    Deleted Parameters:
        common_words (TYPE): Description
    """
    doc2vec = np.array(
        [wv_model[w] for w in token2tfidf.keys() if w in wv_model.vocab])
    doc2vec = doc2vec.sum(axis=0)
    doc2vec /= np.linalg.norm(doc2vec)
    return doc2vec


class Transformer(object):

    """Summary

    Attributes:
        char2vectorizer (TYPE): Description
        common_words (TYPE): Description
        label2index (TYPE): Description
        le (TYPE): Description
        method (TYPE): Description
        min_count (TYPE): Description
        tfidf_vectorizer (TYPE): Description
        w2v_matrix (TYPE): Description
        wv_model (TYPE): Description
    """

    def __init__(self, min_count, method, is_final):
        """
        Args:
            min_count (TYPE): Description
            method (TYPE): Description
            is_final (TYPE): Description
        """
        self.min_count = min_count
        self.method = method

        if self.method == 'c2v':
            self.char2vectorizer = fastText.load_model(
                helper.get_char2vec_model_path(self.min_count, is_final))
        else:
            # load tfidf model
            self.tfidf_vectorizer = TfidfGensimVectorizer(
                dictionary_file=helper.get_dictionary_path(
                    self.min_count, is_final),
                tfidf_model_path=helper.get_tfidf_model_path(
                    self.min_count, is_final)
            )
            logger.info('number of words in tfidf model: %s',
                        len(self.tfidf_vectorizer.token2id.keys()))

            # load w2v model
            self.wv_model = KeyedVectors.load_word2vec_format(
                helper.get_w2v_model_path(self.min_count, is_final))
            logger.info('number of words in wv_model: %s',
                        len(self.wv_model.vocab))
            self.filter_vocabulary()
            self.w2v_matrix = build_w2v_matrix(
                self.wv_model, self.common_words)
            logger.info('w2v_matrix.shape: %s', self.w2v_matrix.shape)

        # encode label
        labels = [str.lower(f.split('_')[0]) for f in CORPUS_FILES]
        logger.info('labels: %s', labels)

        # encode label
        logger.info('encode label')
        self.le = preprocessing.LabelEncoder()
        self.le.fit_transform(labels)

        self.label2index = {l: self.le.transform([l])[0] for l in labels}
        logger.info('encoded label: %s', self.label2index)

    def filter_vocabulary(self):
        """Summary
        """
        tfidf_words = set(self.tfidf_vectorizer.token2id.keys())
        w2v_words = set([w for w in self.wv_model.vocab])
        logger.info(
            'words in w2v and tfidf are the same: %s',
            tfidf_words == w2v_words)
        logger.info('word not in w2v: %s', len(tfidf_words - w2v_words))
        logger.info('word not in tfidf: %s', len(w2v_words - tfidf_words))
        logger.info('word in w2v and tfidf: %s',
                    len(w2v_words & tfidf_words))
        noun_phrases = get_noun_pharses()
        ignored_words = list(noun_phrases) + ['email_token', 'url_token',
                                              'number_token', 'phone_token',
                                              'currency_token',
                                              'datetime_token']
        self.common_words = w2v_words & tfidf_words
        self.common_words = sorted(
            [w for w in self.common_words if w not in ignored_words])
        logger.info('Number of final dictionary: %s', len(self.common_words))

    def encode_labels(self, labels):
        """Summary

        Args:
            labels (TYPE): Description

        Returns:
            like-array: Description
        """
        return self.le.transform(labels)

    def inverse_transform_labels(self, encoded_labels):
        """Summary

        Args:
            encoded_labels (TYPE): Description

        Returns:
            like-array: Description
        """
        return self.le.inverse_transform(encoded_labels)

    def transform(self, document):
        """Summary

        Args:
            document (TYPE): Description

        Returns:
            TYPE: Description
        """
        doc2vec = np.zeros(SKIPGRAM_PARAMS['size'])
        if self.method == 'c2v':
            self.char2vectorizer.get_sentence_vector(document)
        else:
            token2tfidf = self.tfidf_vectorizer.transform(document)

            if len(token2tfidf.keys()) == 0:
                logger.debug('document has no token: %s', document)
            else:
                if self.method == 'w2v':
                    doc2vec = transform_doc2vec_using_w2v(
                        token2tfidf, self.wv_model)
                elif self.method == 'w2v_tfidf':
                    doc2vec = transform_doc2vec_using_w2v_tfidf(
                        token2tfidf, self.w2v_matrix, self.common_words)

        return doc2vec


def convert_preprocessed_text_to_vector(corpus_paths, labels, transformer):
    """Summary

    Args:
        corpus_paths (TYPE): Description
        labels (TYPE): Description
        transformer (TYPE): Description

    Returns:
        TYPE: Description
    """
    logger.info('convert_preprocessed_text_to_vector: %s', locals())
    X = []
    y = []
    encoded_labels = transformer.encode_labels(labels)
    logger.info(
        'encoded_label: %s',
        ','.join(['%s=%s' % (e, l) for e, l in zip(encoded_labels, labels)]))
    for encoded_label, corpus_path in zip(encoded_labels, corpus_paths):
        df = pd.read_csv(corpus_path, header=None, names=['text'])
        y.extend([encoded_label] * df.shape[0])

        logger.info('corpus: %s', corpus_path)
        count = 0
        for index, row in df.iterrows():
            X.append(transformer.transform(row['text']))

            # progress bar
            sys.stdout.write('\r')
            sys.stdout.write('%s/%s' % (index, df.shape[0]))
            sys.stdout.flush()
            count += 1

        sys.stdout.write('\n')
    return np.array(X), np.array(y)


def compute_train_test_matrix(min_count, method, is_final=False):
    """Summary

    Returns:
        TYPE: Description

    Args:
        min_count (TYPE): Description
        method (TYPE): Description
        is_final (bool, optional): Description
    """
    logger.info('prepare train data')

    # load corpus
    logger.info('load corpus and build corpus matrix')
    train_corpus_paths = [helper.get_processed_train_path(DATA_DIR + f,
                                                          is_final)
                          for f in CORPUS_FILES]
    # in the final model, we dont have test dataset
    test_corpus_paths = []
    if is_final is False:
        test_corpus_paths = [helper.get_processed_test_path(DATA_DIR + f)
                             for f in CORPUS_FILES]

    # extract labels from file name
    labels = [str.lower(f.split('_')[0]) for f in CORPUS_FILES]
    logger.info('labels: %s', labels)

    X_train = []
    y_train = []
    X_test = []
    y_test = []
    precomputed_path = helper.get_precomputed_matrix(
        min_count, method, is_final)
    if os.path.isfile(precomputed_path):
        logger.info('load train & test precomputed norm matrix from file')
        data = np.load(precomputed_path)
        X_train = data['X_train']
        y_train = data['y_train']
        X_test = data['X_test']
        y_test = data['y_test']
    else:
        logger.info('build train matrix from corpus')
        transformer = Transformer(min_count, method, is_final)

        X_train, y_train = convert_preprocessed_text_to_vector(
            train_corpus_paths, labels, transformer)
        logger.info('X_train.shape: %s, y_train.shape: %s',
                    X_train.shape, y_train.shape)

        if is_final is False:
            X_test, y_test = convert_preprocessed_text_to_vector(
                test_corpus_paths, labels, transformer)
            logger.info('X_test.shape: %s, y_test.shape: %s',
                        X_test.shape, y_test.shape)

        # save matrix into file
        np.savez(precomputed_path, X_train=X_train, y_train=y_train,
                 X_test=X_test, y_test=y_test)

    return X_train, y_train, X_test, y_test


def cross_validate_logistic_regression(min_count, feature_type):
    """Summary

    Args:
        min_count (TYPE): Description
        feature_type (TYPE): Description
    """
    logger.info(
        'train LogisticRegressionCV model, min_count=%s, extract_method=%s',
        min_count, feature_type)

    X_train, y_train, X_test, y_test = compute_train_test_matrix(
        min_count, feature_type)

    tuned_parameters = [
        {
            'solver': ['newton-cg'],
            'penalty': ['l2'],
            'class_weight': ['balanced', None],
            'multi_class': ['multinomial']},
    ]
    scores = ['precision']

    for score in scores:
        logger.info("# Tuning hyper-parameters for %s", score)
        clf = GridSearchCV(LogisticRegressionCV(Cs=10), tuned_parameters, cv=3,
                           scoring='%s_macro' % score, verbose=2)
        # clf = RandomizedSearchCV(estimator=LogisticRegressionCV(Cs=10),
        #                          param_distributions=tuned_parameters,
        #                          n_iter=20, scoring='%s_macro' % score,
        #                          cv=3, verbose=2,
        #                          random_state=42, n_jobs=-1)

        clf.fit(X_train, y_train)

        logger.info("Best parameters set found on development set:")
        logger.info(clf.best_params_)
        logger.info("Grid scores on development set:")
        means = clf.cv_results_['mean_test_score']
        stds = clf.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, clf.cv_results_['params']):
            logger.info("%0.3f (+/-%0.03f) for %r", mean, std * 2, params)

        logger.info("Detailed classification report:")
        logger.info("The model is trained on the full development set.")
        logger.info("The scores are computed on the full evaluation set.")
        y_true, y_pred = y_test, clf.predict(X_test)
        logger.info(classification_report(y_true, y_pred))
        logger.info('Confusion_matrix: \n%s',
                    confusion_matrix(y_true, y_pred))


def cross_validate_adaboost(min_count, feature_type):
    """Summary

    Args:
        min_count (TYPE): Description
        feature_type (TYPE): Description
    """

    logger.info(
        'train AdaBoostClassifier model, min_count=%s, extract_method=%s',
        min_count, feature_type)

    X_train, y_train, X_test, y_test = compute_train_test_matrix(
        min_count, feature_type)

    n_estimators = [int(x) for x in np.linspace(50, 200, num=10)]
    tuned_parameters = {
        'n_estimators': n_estimators,
        'algorithm': ['SAMME', 'SAMME.R'],
    }

    scores = ['precision']

    # train AdaBoostClassifier
    for score in scores:
        logger.info("# Tuning hyper-parameters for %s", score)

        clf = RandomizedSearchCV(estimator=AdaBoostClassifier(),
                                 param_distributions=tuned_parameters,
                                 n_iter=20, scoring='%s_macro' % score,
                                 cv=3, verbose=2,
                                 random_state=42, n_jobs=-1)
        clf.fit(X_train, y_train)

        logger.info("Best parameters set found on development set:")
        logger.info(clf.best_params_)
        logger.info("Grid scores on development set:")
        means = clf.cv_results_['mean_test_score']
        stds = clf.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, clf.cv_results_['params']):
            logger.info("%0.3f (+/-%0.03f) for %r", mean, std * 2, params)

        logger.info("Detailed classification report:")
        logger.info("The model is trained on the full development set.")
        logger.info("The scores are computed on the full evaluation set.")
        y_true, y_pred = y_test, clf.predict(X_test)
        logger.info(classification_report(y_true, y_pred))
        logger.info('Confusion_matrix: \n%s',
                    confusion_matrix(y_true, y_pred))


def cross_validate_knn(min_count, feature_type):
    """
    Args:
        min_count (TYPE): Description
        feature_type (TYPE): Description

    """
    logger.info(
        'train KNN model, min_count=%s, extract_method=%s',
        min_count, feature_type)

    X_train, y_train, X_test, y_test = compute_train_test_matrix(
        min_count, feature_type)

    n_neighbors = [int(x) for x in np.linspace(5, 20, num=3)]
    tuned_parameters = {
        'n_neighbors': n_neighbors,
        'algorithm': ['kd_tree'],
    }

    scores = ['precision']

    # train KNN
    for score in scores:
        logger.info("# Tuning hyper-parameters for %s", score)

        clf = GridSearchCV(KNeighborsClassifier(), tuned_parameters, cv=3,
                           scoring='%s_macro' % score, verbose=2, n_jobs=-1)
        clf.fit(X_train, y_train)

        logger.info("Best parameters set found on development set:")
        logger.info(clf.best_params_)
        logger.info("Grid scores on development set:")
        means = clf.cv_results_['mean_test_score']
        stds = clf.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, clf.cv_results_['params']):
            logger.info("%0.3f (+/-%0.03f) for %r", mean, std * 2, params)

        logger.info("Detailed classification report:")
        logger.info("The model is trained on the full development set.")
        logger.info("The scores are computed on the full evaluation set.")
        y_true, y_pred = y_test, clf.predict(X_test)
        logger.info(classification_report(y_true, y_pred))
        logger.info('Confusion_matrix: \n%s',
                    confusion_matrix(y_true, y_pred))


def cross_validate_randomforest(min_count, feature_type):
    """Summary

    Args:
        min_count (TYPE): Description
        feature_type (TYPE): Description
    """
    logger.info(
        'train RandomForestClassifier model, min_count=%s, extract_method=%s',
        min_count, feature_type)

    X_train, y_train, X_test, y_test = compute_train_test_matrix(
        min_count, feature_type)

    # params = {'criterion': 'gini',
    #           'min_samples_split': 5,
    #           'n_estimators': 200,
    #           'min_samples_leaf': 1,
    #           'max_depth': 110,
    #           'max_features': 'auto',
    #           'bootstrap': False}
    # clf = RandomForestClassifier(**params, random_state=42)
    # logger.info(clf.get_params())
    # clf.fit(X_train, y_train)

    # # train accuracy
    # logger.info('TRAIN RESULT:')
    # y_train_pred = clf.predict(X_train)
    # accuracy = np.mean(y_train_pred.ravel() == y_train.ravel()) * 100
    # logger.info('Accuracy: %.1f', accuracy)
    # logger.info('Confusion_matrix: \n%s',
    #             confusion_matrix(y_train, y_train_pred))

    # logger.info('TEST RESULT:')
    # y_test_pred = clf.predict(X_test)
    # logger.info('Accuracy: %.1f', accuracy_score(y_test, y_test_pred) * 100)

    # logger.info('Classification report: \n%s',
    #             classification_report(y_test, y_test_pred))

    # logger.info('Confusion_matrix: \n%s',
    #             confusion_matrix(y_test, y_test_pred))

    n_estimators = [int(x) for x in np.linspace(start=200, stop=5000, num=10)]
    max_features = ['auto', 'sqrt', 'log2']
    max_depth = [int(x) for x in np.linspace(100, 1000, num=5)]
    max_depth.append(None)
    min_samples_split = [2, 5, 10]
    min_samples_leaf = [1, 3, 10]
    bootstrap = [True, False]
    criterion = ["gini", "entropy"]

    tuned_parameters = {'n_estimators': n_estimators,
                        'max_features': max_features,
                        'max_depth': max_depth,
                        'min_samples_split': min_samples_split,
                        'min_samples_leaf': min_samples_leaf,
                        'bootstrap': bootstrap,
                        "criterion": criterion}

    scores = ['precision']

    for score in scores:
        logger.info("# Tuning hyper-parameters for %s", score)

        clf = GridSearchCV(RandomForestClassifier(), tuned_parameters, cv=5,
                           scoring='%s_macro' % score, n_jobs=-1, verbose=2)
        # clf = RandomizedSearchCV(
        #     estimator=RandomForestClassifier(),
        #     param_distributions=tuned_parameters,  n_iter=20,
        #     scoring='%s_macro' % score, cv=3, verbose=2, random_state=42,
        #     n_jobs=-1)
        clf.fit(X_train, y_train)

        logger.info("Best parameters set found on development set:")
        logger.info(clf.best_params_)
        logger.info("Grid scores on development set:")
        means = clf.cv_results_['mean_test_score']
        stds = clf.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, clf.cv_results_['params']):
            logger.info("%0.3f (+/-%0.03f) for %r", mean, std * 2, params)

        logger.info("Detailed classification report:")
        logger.info("The model is trained on the full development set.")
        logger.info("The scores are computed on the full evaluation set.")
        y_true, y_pred = y_test, clf.predict(X_test)
        logger.info(classification_report(y_true, y_pred))
        logger.info('Confusion_matrix: \n%s',
                    confusion_matrix(y_true, y_pred))


def cross_validate_gmm(min_count, feature_type):
    """Summary

    Args:
        min_count (TYPE): Description
        feature_type (TYPE): Description
    """
    logger.info(
        'train GaussianMixture model, min_count=%s, extract_method=%s',
        min_count, feature_type)

    X_train, y_train, X_test, y_test = compute_train_test_matrix(
        min_count, feature_type)

    cov_types = ['full', 'tied', 'diag', 'spherical']
    n_components = 3
    for cov_type in cov_types:
        logger.info('cov_type: %s', str.upper(cov_type))

        clf = GaussianMixture(n_components=n_components,
                              covariance_type=cov_type, max_iter=100,
                              random_state=42, verbose=2)

        # Since we have class labels for the training data, we can
        # initialize the GMM parameters in a supervised manner.
        clf.means_init = np.array(
            [X_train[y_train == i].mean(axis=0)
             for i in range(n_components)])

        # Train the other parameters using the EM algorithm.
        clf.fit(X_train)

        logger.info('TRAIN RESULT:')
        y_train_pred = clf.predict(X_train)
        accuracy = np.mean(y_train_pred.ravel() == y_train.ravel()) * 100
        logger.info('Accuracy: %.1f', accuracy)
        logger.info('Confusion_matrix: \n%s',
                    confusion_matrix(y_train, y_train_pred))

        logger.info('TEST RESULT:')
        y_test_pred = clf.predict(X_test)
        logger.info('Accuracy: %.1f', accuracy_score(
            y_test, y_test_pred) * 100)

        logger.info('Classification report: \n%s',
                    classification_report(y_test, y_test_pred))

        logger.info('Confusion_matrix: \n%s',
                    confusion_matrix(y_test, y_test_pred))


def cross_validate(args):
    """Summary

    Args:
        args (TYPE): Description
    """
    if args.model == 'knn':
        cross_validate_knn(args.min_count, args.feature_type)
    elif args.model == 'adb':
        cross_validate_adaboost(args.min_count, args.feature_type)
    elif args.model == 'lr':
        cross_validate_logistic_regression(args.min_count, args.feature_type)
    elif args.model == 'rf':
        cross_validate_randomforest(args.min_count, args.feature_type)
    elif args.model == 'gmm':
        cross_validate_gmm(args.min_count, args.feature_type)


def train(args):
    """
    Train the final model

    Args:
        args (TYPE): Description

    Deleted Parameters:
        model_type (TYPE): Description
    """
    logger.info(
        'train final model, min_count=%s, extract_method=%s',
        args.min_count, args.feature_type)

    # preprocess all data
    preprocess.preprocess(CORPUS_FILES, skip_viTokenizer=False,
                          export_pos=False, test_size=0., is_final=True)

    # build word2vec model
    w2v.train_model(min_count=args.min_count,
                    use_external_data=False, is_final=True)

    # build tfidf model
    tfidf.train_tfidf(min_count=args.min_count, is_final=True)

    X_train, y_train, _, _ = compute_train_test_matrix(
        args.min_count, args.feature_type, is_final=True)

    params = {'criterion': 'gini',
              'min_samples_split': 5,
              'n_estimators': 200,
              'min_samples_leaf': 1,
              'max_depth': 110,
              'max_features': 'auto',
              'bootstrap': False}
    clf = RandomForestClassifier(**params, random_state=42, verbose=2,
                                 n_jobs=-1)
    clf.fit(X_train, y_train)
    joblib.dump(clf, helper.get_model_path('RandomForest', is_final=True))


def predict_file(args):
    """Summary

    Args:
        args (TYPE): Description

    Returns:
        TYPE: Description

    Deleted Parameters:
        input_path (TYPE): Description
    """
    print('args: ', args)
    print('download nltk (words, punkt) if it does not')
    helper.prepare_nltk()

    min_count = 5
    transform_method = 'w2v'
    model_type = 'RandomForest'
    print('Model: %s, min count of term: %s, feature extraction: %s' % (
        model_type, min_count, transform_method))

    input_path = args.input_path
    if os.path.isfile(input_path) is False:
        print(
            'Input file "%s" does not exist. Please check again.' % input_path)
        return
    tqdm.pandas()

    print('\nLoad test file')
    df = pd.read_csv(input_path, header=None, names=['text'])
    print('\tdata size: %s' % df.shape[0])

    print('Start cleaning...')
    df = df.dropna()
    print('\tdata size after dropna: %s' % df.shape[0])

    print('\tpreprocess data...')
    clean_text_series = df.progress_apply(
        lambda x: preprocess.preprocess_document(
            x['text'], skip_viTokenizer=False, export_pos=False),
        axis=1)

    print('Feature extraction...')
    transformer = Transformer(min_count, transform_method, is_final=True)
    X = []
    for index, clean_text in clean_text_series.iteritems():
        X.append(transformer.transform(clean_text))

        # progress bar
        sys.stdout.write('\r')
        sys.stdout.write('%s/%s' % (index, len(clean_text_series)))
        sys.stdout.flush()
    sys.stdout.write('\n')

    X = np.array(X)
    print('\tX.shape: ', X.shape)

    print('Load model and predict...')
    clf = joblib.load(helper.get_model_path(model_type, is_final=True))
    y_pred = clf.predict(X)

    result = pd.DataFrame.from_dict(
        {'label': transformer.inverse_transform_labels(y_pred),
         'text': df.text.tolist()})

    print('Export result to file: %s' % args.output_path)
    result.to_csv(args.output_path, index=False,
                  header=False, columns=['label', 'text'])
