"""Summary
"""
import matplotlib as mpl
mpl.use('Agg')
from gensim.models import KeyedVectors
import argparse
from src import viz_helper
import logging


logging.basicConfig(
    # filename='log_preprocess.log',
    format='%(asctime)s : %(levelname)s : %(message)s',
    level=logging.INFO)

parser = argparse.ArgumentParser()

if __name__ == '__main__':
    # wv_ex_model = KeyedVectors.load_word2vec_format(
    #     'model/sentiment_external_sg.vec')

    parser.add_argument('type', choices=['tsne', 'mds'],
                        help='choose type of visualization')
    parser.add_argument(
        '-nw', '--number_words', type=int,
        help='number of words to visualize',
        default=None)

    args = parser.parse_args()
    print('load word2vec model')
    wv_model = KeyedVectors.load_word2vec_format('model/sentiment_sg.vec')
    if args.type == 'tsne':
        print('plot tsne_sentiment_sg')
        viz_helper.tsne_plot(wv_model, 'output/tsne_sentiment_sg.png')

    elif args.type == 'mds':
        print('plot MDS')
        viz_helper.mds_plot(
            wv_model, 'output/mds_sentiment_sg.html',
            nb_words=args.number_words)
