from sklearn.cluster import AffinityPropagation
from sklearn import metrics
import numpy as np
import gensim
import time

model_path = 'model/doc2vec.vec'
model = gensim.models.doc2vec.Doc2Vec.load(model_path)

print('Number of documents', len(model.docvecs))
step = 10000
durations = []
num_clusters = []
total_docs = len(model.docvecs)
for i in range(0, total_docs, step):
    start_time = time.time()
    stop = i + step if (i + step) < total_docs else total_docs
    print('\nRANGE: %s -> %s' % (i, stop))
    X = np.array([model.docvecs[k] for k in range(i, stop)])
    print('X.shape=', X.shape)

    clf = AffinityPropagation(verbose=2).fit(X)
    y_pred = clf.labels_
    num_cluster = len(set(y_pred))
    num_clusters.append(num_cluster)
    print('Number of cluster:', num_cluster)

    # evaluate
    print('silhouette_score:', metrics.silhouette_score(
        X, y_pred, metric='euclidean'))

    # duration
    elapsed_time = time.time() - start_time
    print('duration: ', time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))
    durations.append(elapsed_time)

print('\nAverage durations: %s' % np.average(durations))
print('clusters: %s' % num_clusters)
print('Average num_cluster: %s' % np.average(num_clusters))
