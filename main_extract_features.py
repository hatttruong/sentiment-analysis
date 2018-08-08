import logging
import argparse

from src import w2v
from src import tfidf
from src import doc2vec

logging.basicConfig(
    # filename='log/log_extract_features.log',
    format='%(asctime)s : %(levelname)s : %(message)s',
    level=logging.INFO)
parser = argparse.ArgumentParser()


if __name__ == '__main__':
    parser.add_argument('type', choices=['w2v', 'tfidf', 'c2v', 'd2v'],
                        help='choose type of extracting features')
    parser.add_argument('-e', '--external_corpus', action='store_true',
                        help='include external_corpus')
    parser.add_argument('min_count', type=int,
                        help='min count of token to be kept')
    parser.add_argument(
        '-fn', '--is_final', action='store_true',
        help='preprocess for final model (not split to train/test)')
    parser.add_argument(
        '-ex', '--use_external_data', action='store_true',
        help='use external data (use data from organizer)')

    args = parser.parse_args()
    if args.type == 'w2v':
        w2v.train_model(
            args.min_count, use_external_data=args.external_corpus,
            is_final=args.is_final)
    elif args.type == 'tfidf':
        tfidf.train_tfidf(min_count=args.min_count, is_final=args.is_final)
    elif args.type == 'd2v':
        doc2vec.train_model(
            use_extend=args.use_external_data, is_final=args.is_final)
