import logging
import argparse

from src import preprocess
from src.constants import *

logging.basicConfig(
    filename='log/log_preprocess.log',
    format='%(asctime)s : %(levelname)s : %(message)s',
    level=logging.INFO)
parser = argparse.ArgumentParser()


if __name__ == '__main__':
    parser.add_argument('corpus', choices=['def', 'ex', 'both'],
                        help='choose type of corpus to preprocess')
    parser.add_argument('-pos', '--export_pos', action='store_true',
                        help='export postagging')
    parser.add_argument(
        '-fn', '--is_final', action='store_true',
        help='preprocess for final model (not split to train/test)')

    args = parser.parse_args()
    if args.corpus in ['def', 'both']:
        preprocess.preprocess(CORPUS_FILES, skip_viTokenizer=False,
                              export_pos=args.export_pos,
                              is_final=args.is_final)
    if args.corpus in ['ex', 'both']:
        preprocess.preprocess(EXTERNAL_CORPUS_FILES, skip_viTokenizer=True,
                              export_pos=args.export_pos,
                              is_final=args.is_final)
