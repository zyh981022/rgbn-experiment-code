from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

data = mnist.train.next_batch(60000)
#points = np.load("points.npy")
points = data[0]

from sklearn.cluster import SpectralClustering

sc = SpectralClustering(n_clusters = 101, assign_labels="kmeans")

max_points = 10000

sc.fit(points[:max_points])

print(max(sc.labels_))
labels = sc.labels_
centroids = []

empty = []
for i in range(100):
    if labels[labels == i].shape[0] == 0:
        continue
    centroids.append(points[:max_points][labels == i].sum(axis = 0) / labels[labels == i].shape[0])
    print(labels[labels == i].shape[0])

print(len(centroids))

centroids = np.array(centroids)

np.save("sc_centroids.npy", centroids)

