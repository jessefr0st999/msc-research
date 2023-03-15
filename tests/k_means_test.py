import random

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from helpers import configure_plots, get_map, prepare_df

COLOURS = ['red', 'green', 'blue', 'gold', 'magenta', 'orange', 'cyan', 'saddlebrown']

n_clusters = 4
points = [
    (1, 1), (2, 1), (1, 1.5),
    (3, 3), (4, 4), (5, 3), (5, 4),
    (6, 4), (7, 3), (7, 4), (8, 8),
    (0, 6), (0, 6.5), (0.5, 6), (0.5, 6.5),
]

# kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init='auto').fit(points)
# clusters = list(kmeans.labels_)

# Optimal (k-means)
# clusters = [2, 2, 2, 3, 3, 3, 3, 1, 1, 1, 1, 0, 0, 0, 0]

# Sub-optimal
# clusters = [2, 2, 2, 3, 3, 3, 3, 1, 1, 1, 1, 0, 3, 0, 0]

# Poor
clusters = [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 0, 1, 2, 3]

# Random
# clusters = [random.randrange(n_clusters) for _ in points]

sil_score = silhouette_score(points, clusters)
print(clusters)
print(sil_score)

figure, axis = plt.subplots(1)
x_vec, y_vec = zip(*points)
node_colours = [COLOURS[i] for i in clusters]
axis.scatter(x_vec, y_vec, c=node_colours, s=150)
plt.show()