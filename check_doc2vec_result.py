import gensim
import pandas as pd
import numpy as np
import logging
import operator

logger = logging.getLogger(__name__)


def iterdocuments(filenames, tokens_only=False):
    """
    Iterate over documents, yielding a list of utf8 tokens at a time.

    Args:
        filenames (TYPE): Description
        encoding (str, optional): Description

    Yields:
        TYPE: Description
    """
    index = 0
    for filename in filenames:
        df = pd.read_csv(filename, names=['text'])
        for _, row in df.iterrows():
            if tokens_only:
                yield gensim.utils.simple_preprocess(row['text'])
            else:
                yield gensim.models.doc2vec.TaggedDocument(
                    gensim.utils.simple_preprocess(row['text']), [index])
                index += 1


def main():
    models = {
        'dm_concat': None,
        'dbow_concat': None,
        'dm_sum': None,
        'dbow_sum': None,
        'dm_mean': None,
        'dbow_mean': None
    }
    externals = [True, False]
    pattern = 'model/doc2vec/doc2vec%s_%s.vec'

    for ex in externals:
        data_path = 'data/sentences_data.csv'
        if ex:
            data_path = 'data/sentences_data_extend.csv'

        logger.info('load corpus: %s', data_path)
        train_corpus = list(iterdocuments([data_path]))
        logger.info('number of data: %s', len(train_corpus))

        external = '_external' if ex else ''
        for k in models.keys():
            models[k] = pattern % (external, k)
            model = gensim.models.doc2vec.Doc2Vec.load(models[k])
            logger.info('MODEL: %s', k)
            logger.info('Number of documents: %s', len(model.docvecs))

            # get similarity by words
            compared_words = ['ram', 'bộ_nhớ', 'dung_lượng']
            for w in compared_words:
                logger.info('Compare with "%s"', w.upper())
                similars = model.wv.similar_by_word(w, topn=10)
                logger.info('\t' + ' '.join(['(%s, %.2f)' %
                                             (x[0], x[1]) for x in similars]))

            # get similar document
            compared_document_ids = np.random.randint(
                1, high=len(train_corpus), size=3)
            for doc_id in compared_document_ids:
                inferred_vector = model.infer_vector(
                    train_corpus[doc_id].words)
                sims = model.docvecs.most_similar(
                    [inferred_vector], topn=len(model.docvecs))
                logger.info('Document ({}): «{}»'.format(
                    doc_id, ' '.join(train_corpus[doc_id].words)))
                logger.info(u'SIMILAR/DISSIMILAR DOCS PER MODEL %s:', model)
                for index in range(20):
                    logger.info(u'%s: «%s»', sims[index],
                                ' '.join(train_corpus[sims[index][0]].words))

            # check with aspects


def check_aspect():
    """Summary
    """
    file_path = 'data/features_dien-thoai-may-tinh-bang_affinity_token_set_ratio.csv'
    df = pd.read_csv(file_path)
    group_features = list()
    for index, row in df.iterrows():
        temp_features = row["similar_features"][1:-1].split(',')
        temp_features = [x.strip("'. ") for x in temp_features]
        group_features.append(set([x for x in temp_features if len(x) > 0]))

    models = {
        'dm_concat': None,
        'dbow_concat': None,
        'dm_sum': None,
        'dbow_sum': None,
        'dm_mean': None,
        'dbow_mean': None
    }
    externals = [True, False]
    pattern = 'model/doc2vec/doc2vec%s_%s.vec'

    compared_group_ids = np.random.randint(
        0, high=len(group_features), size=20)
    for ex in externals:
        data_path = 'data/sentences_data.csv'
        if ex:
            data_path = 'data/sentences_data_extend.csv'

        logger.info('load corpus: %s', data_path)
        train_corpus = list(iterdocuments([data_path]))
        logger.info('number of data: %s', len(train_corpus))

        external = '_external' if ex else ''
        for k in models.keys():
            models[k] = pattern % (external, k)
            model = gensim.models.doc2vec.Doc2Vec.load(models[k])
            logger.info('MODEL: %s', k)

            # for each group features, get the most similar document
            for idx in compared_group_ids:
                logger.info('group_features(%s): «%s«', idx,
                            ', '.join(group_features[idx]))
                sim_documents = []
                for feature in group_features[idx]:
                    inferred_vector = model.infer_vector(
                        gensim.utils.simple_preprocess(feature))
                    sims = model.docvecs.most_similar(
                        [inferred_vector], topn=len(model.docvecs))
                    sim_documents.extend(sims[:20])

                sim_documents = sorted(
                    sim_documents, key=operator.itemgetter(1), reverse=True)
                not_duplicated = dict()
                for sim_document in sim_documents:
                    if sim_document[0] not in not_duplicated.keys():
                        not_duplicated[sim_document[0]] = sim_document
                sim_documents = not_duplicated.values()
                # sort again
                sim_documents = sorted(
                    sim_documents, key=operator.itemgetter(1), reverse=True)

                logger.info(u'SIMILAR/DISSIMILAR DOCS PER MODEL %s:', k)
                for index in range(10):
                    logger.info(
                        u'%s: «%s»', sim_documents[index],
                        ' '.join(train_corpus[sim_documents[index][0]].words))


logging.basicConfig(
    filename='log/check_get_document_similar_with_aspect.log',
    format='%(asctime)s : %(levelname)s : %(message)s',
    level=logging.INFO)


if __name__ == '__main__':
    # main()
    check_aspect()
