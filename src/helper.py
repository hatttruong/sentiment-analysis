"""Summary
"""
import numpy as np
import os
import nltk

from src.constants import *


def get_processed_path(file_path):
    """Summary

    Args:
        file_path (TYPE): Description

    Returns:
        TYPE: Description
    """
    return file_path + '.preprocessed'


def get_processed_train_path(file_path, is_final):
    """Summary

    Args:
        file_path (TYPE): Description

    Returns:
        TYPE: Description
    """
    if is_final:
        return get_processed_path(file_path)
    else:
        return get_processed_path(file_path) + '.train'


def get_processed_test_path(file_path):
    """Summary

    Args:
        file_path (TYPE): Description

    Returns:
        TYPE: Description
    """
    return get_processed_path(file_path) + '.test'


def get_w2v_model_path(min_count, is_final):
    """Summary

    Returns:
        TYPE: Description

    Args:
        min_count (TYPE): Description
    """
    if is_final:
        return os.path.join(MODEL_DIR,
                            'skipgram_min_count_%s_final.vec' % min_count)
    else:
        return os.path.join(MODEL_DIR,
                            'skipgram_min_count_%s.vec' % min_count)


def get_doc2vec_model_path(type_model, use_external, is_final):
    """Summary

    Returns:
        TYPE: Description

    Args:
        type_model (TYPE): Description
        use_external (TYPE): Description
        is_final (TYPE): Description

    """
    external = '_external' if use_external else ''
    if is_final:
        return os.path.join(
            MODEL_DIR,
            'doc2vec%s_%s_final.vec' % (external, type_model))
    else:
        return os.path.join(
            MODEL_DIR,
            'doc2vec%s_%s.vec' % (external, type_model))


def get_external_w2v_model_path():
    """Summary

    Returns:
        TYPE: Description
    """
    return os.path.join(MODEL_DIR, 'skipgram_external.vec')


def get_tfidf_model_path(min_count, is_final):
    """Summary

    Returns:
        TYPE: Description

    Args:
        min_count (TYPE): Description
    """
    if is_final:
        return os.path.join(MODEL_DIR,
                            'tfidf_min_count_%s_final.model' % min_count)
    else:
        return os.path.join(MODEL_DIR,
                            'tfidf_min_count_%s.model' % min_count)


def get_char2vec_model_path(min_count, is_final):
    """Summary

    Returns:
        TYPE: Description

    Args:
        min_count (TYPE): Description
    """
    if is_final:
        return os.path.join(MODEL_DIR,
                            'char2vec_min_count_%s_final.vec' % min_count)
    else:
        return os.path.join(MODEL_DIR,
                            'char2vec_min_count_%s.vec' % min_count)


def get_dictionary_path(min_count, is_final):
    """Summary

    Returns:
        TYPE: Description
    """
    if is_final:
        return os.path.join(MODEL_DIR,
                            'sentiment_min_count_%s_final.dict' % min_count)
    else:
        return os.path.join(MODEL_DIR,
                            'sentiment_min_count_%s.dict' % min_count)


def get_model_path(model_type, is_final):
    """Summary

    Returns:
        TYPE: Description

    Args:
        model_type (TYPE): Description
    """
    if is_final:
        return os.path.join(MODEL_DIR, 'sentiment_%s_final.model' % model_type)
    else:
        return os.path.join(MODEL_DIR, 'sentiment_%s.model' % model_type)


def get_precomputed_matrix(min_count, method, is_final):
    """Summary

    Returns:
        TYPE: Description
    """
    if is_final:
        return os.path.join(
            MODEL_DIR,
            'train_test_pre_norm_matrix_mincount_%s_%s_final.npz' % (
                min_count, method)
        )
    else:
        return os.path.join(
            MODEL_DIR,
            'train_test_pre_norm_matrix_mincount_%s_%s.npz' % (
                min_count, method)
        )


def load_data_and_labels(positive_data_file, negative_data_file,
                         neutral_data_file):
    """
    Loads data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    positive_examples = list(
        open(positive_data_file, "r", encoding='utf-8').readlines())
    positive_examples = [s.strip() for s in positive_examples]
    negative_examples = list(
        open(negative_data_file, "r", encoding='utf-8').readlines())
    negative_examples = [s.strip() for s in negative_examples]
    neutral_examples = list(
        open(neutral_data_file, "r", encoding='utf-8').readlines())
    neutral_examples = [s.strip() for s in neutral_examples]

    # Split by words
    x_text = positive_examples + negative_examples + neutral_examples
    # x_text = [clean_str(sent) for sent in x_text]

    # Generate labels
    positive_labels = [[0, 0, 1] for _ in positive_examples]
    negative_labels = [[1, 0, 0] for _ in negative_examples]
    neutral_labels = [[0, 1, 0] for _ in neutral_examples]
    y = np.concatenate([positive_labels, negative_labels, neutral_labels], 0)
    return [x_text, y]


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data) - 1) / batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]


def prepare_nltk():
    """Summary
    """
    nltk.download('words')
    nltk.download('punkt')
