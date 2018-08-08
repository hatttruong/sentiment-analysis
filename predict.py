"""
Your model will receive ONE .csv file of new posts (our test set), in the same
format of Negative train.csv or Neutral train.csv or Positive train.csv.

The expected output file that your model returns will be a .csv file in the
same format of sample.csv file

"""
import logging
import argparse


from src.classify import *

logging.basicConfig(
    filename='log/log_predict.log',
    format='%(asctime)s : %(levelname)s : %(message)s',
    level=logging.INFO)

parser = argparse.ArgumentParser()

if __name__ == '__main__':
    parser.add_argument('input_path', help='input path')
    parser.add_argument('output_path', help='output path')

    args = parser.parse_args()
    predict_file(args)
