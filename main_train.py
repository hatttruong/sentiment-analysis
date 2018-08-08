import logging
import argparse

from src import classify

logging.basicConfig(
    filename='log/log_train_rf.log',
    format='%(asctime)s : %(levelname)s : %(message)s',
    level=logging.INFO)
parser = argparse.ArgumentParser()


if __name__ == '__main__':
    parser.add_argument(
        'action', choices=['cv', 't'],
        help='cross-validation or train model'
    )
    parser.add_argument(
        'model', choices=['knn', 'gmm', 'adb', 'lr', 'rf'],
        help='Chose model to train: KNN, Gaussian Mixture, AdaBoost, Logistic Regression CV, RandomForest'
    )
    parser.add_argument('feature_type', choices=['w2v', 'tfidf', 'c2v'],
                        help='choose type of extracting features')
    parser.add_argument('min_count', type=int,
                        help='min count of token to be kept')

    args = parser.parse_args()
    if args.action == 't':
        classify.train(args)
    elif args.action == 'cv':
        classify.cross_validate(args)
