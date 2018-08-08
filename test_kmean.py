from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import KMeans
from sklearn import metrics
import numpy as np
import gensim
import time
import operator

model_path = 'model/doc2vec.vec'
model = gensim.models.doc2vec.Doc2Vec.load(model_path)

print('Number of documents', len(model.docvecs))
total_docs = len(model.docvecs)
X = np.array([model.docvecs[i] for i in range(20000)])
print('X.shape=', X.shape)

calinski_harabaz_scores = dict()
silhouette_scores = dict()
for i in range(50, 100):
    start_time = time.time()
    kmeans_model = MiniBatchKMeans(
        n_clusters=i, random_state=1, verbose=2).fit(X)
    # kmeans_model = KMeans(n_clusters=i, random_state=1,
    #                       n_jobs=-1, verbose=2).fit(X)
    labels = kmeans_model.labels_

    # evaluate
    calinski_harabaz_score = metrics.calinski_harabaz_score(X, labels)
    silhouette_score = metrics.silhouette_score(X, labels, metric='euclidean')
    print('calinski_harabaz_score:', calinski_harabaz_score)
    print('silhouette_score:', silhouette_score)
    calinski_harabaz_scores[i] = calinski_harabaz_score
    silhouette_scores[i] = silhouette_score
    # duration
    elapsed_time = time.time() - start_time
    print('duration: ', time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))


# sort by silhouette_score
sorted_scores = sorted(
    silhouette_scores.items(), key=operator.itemgetter(1), reverse=True)
print('sort by silhouette_score: ', sorted_scores[:5])

# sort by calinski_harabaz_score
sorted_scores = sorted(
    calinski_harabaz_scores.items(), key=operator.itemgetter(1), reverse=True)
print('sort by calinski_harabaz_score: ', sorted_scores[:5])
