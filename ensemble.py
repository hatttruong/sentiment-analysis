"""Summary

Attributes:
    logger (TYPE): Description
"""
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix

import logging

from src import helper


logger = logging.getLogger(__name__)


def main():
    """Summary
    """
    file_paths = [helper.get_precomputed_matrix(5, 'w2v', False),
                  helper.get_precomputed_matrix(5, 'w2v_tfidf', False),
                  helper.get_precomputed_matrix(1, 'w2v', False),
                  helper.get_precomputed_matrix(1, 'w2v_tfidf', False)]
    for file_path in file_paths:
        logger.info('file: %s', file_path)
        data = np.load(file_path)
        X_train = data['X_train']
        y_train = data['y_train']
        # X_test = data['X_test']
        # y_test = data['y_test']

        rf_params = {'criterion': 'gini',
                     'min_samples_split': 5,
                     'n_estimators': 200,
                     'min_samples_leaf': 1,
                     'max_depth': 110,
                     'max_features': 'auto',
                     'bootstrap': False}
        clf1 = RandomForestClassifier(
            **rf_params, random_state=42)
        clf2 = AdaBoostClassifier(
            **{'n_estimators': 200, 'algorithm': 'SAMME.R'})
        clf3 = KNeighborsClassifier(
            **{'algorithm': 'kd_tree', 'n_neighbors': 12})
        clf4 = LogisticRegressionCV(
            Cs=10, **{'class_weight': None, 'penalty': 'l2',
                      'solver': 'newton-cg', 'multi_class': 'multinomial'})
        clf5 = GaussianNB()
        eclf_hard = VotingClassifier(
            estimators=[('rf', clf1), ('ab', clf2), ('knn', clf3),
                        ('lr', clf4), ('gnb', clf5)],
            voting='hard')
        eclf_soft = VotingClassifier(
            estimators=[('rf', clf1), ('ab', clf2), ('knn', clf3),
                        ('lr', clf4), ('gnb', clf5)],
            voting='soft')
        for clf, label in zip(
            [clf1, clf2, clf3, clf4, clf5, eclf_hard, eclf_soft],
            ['Random Forest', 'AdaBoost', 'KNN', 'Logistic Regression',
             'naive Bayes', 'Ensemble Hard', 'Ensemble Soft']):
            logger.info('Algorithm: %s', label)
            scores = cross_val_score(
                clf, X_train, y_train, cv=5, scoring='accuracy')

            logger.info('CROSS VALIDATION:')
            logger.info("Accuracy: %0.2f (+/- %0.2f) [%s]",
                        scores.mean(), scores.std(), label)

        # logger.info('TEST RESULT:')
        # y_test_pred = clf.predict(X_test)
        # accuracy = np.mean(y_test_pred.ravel() == y_test.ravel()) * 100
        # logger.info('Accuracy: %.1f', accuracy)
        # logger.info('Confusion_matrix: \n%s',
        #             confusion_matrix(y_test, y_test_pred))


logging.basicConfig(
    filename='log/log_ensemble.log',
    format='%(asctime)s : %(levelname)s : %(message)s',
    level=logging.INFO)

if __name__ == '__main__':
    main()
