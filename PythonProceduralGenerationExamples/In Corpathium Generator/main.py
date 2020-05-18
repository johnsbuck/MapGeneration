import numpy as np
import matplotlib.pyplot as plt
from matplotlib import collections as mc
from sklearn.neighbors import NearestNeighbors
from Utilities.metric import euclidean
from Utilities.geometry import line_segment_intersection, is_between
from RandomPointGenerator.PoissonDiscSampleGenerator import PoissonDiscSampleGenerator


# np.random.seed(1037)
# X = np.random.uniform(-25, 25, (12, 2))
X = np.random.permutation(PoissonDiscSampleGenerator(extent=[50, 50]).generate())[:50]

n_neighbors = 5
neighbors = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean").fit(X)
distances, indices = neighbors.kneighbors(X)
indices = list(indices)

for i in range(len(indices)):
    new_size = np.random.randint(2, 6)
    indices[i] = list(indices[i][:min(new_size, len(indices[i]))])
    print(len(indices[i]))

lines = []
i = 0
for neighs in indices:
    for n in range(len(neighs)):
        ignore = False

        lindex = 0
        while lindex < len(lines):
            # Check if an intersection exists
            if line_segment_intersection(lines[lindex][0], lines[lindex][1], X[i], X[neighs[n]]):
                # Check if one point is between two point
                if is_between(lines[lindex][0], lines[lindex][1], X[i]) or \
                   is_between(lines[lindex][0], lines[lindex][1], X[neighs[n]]) or \
                   is_between(X[i], X[neighs[n]], lines[lindex][0]) or \
                   is_between(X[i], X[neighs[n]], lines[lindex][1]):
                    lindex += 1
                    continue

                if euclidean(lines[lindex][0], lines[lindex][1]) <= euclidean(X[i], X[neighs[n]]):
                    ignore = True
                    break
                else:
                    lines.pop(lindex)
            else:
                lindex += 1

        if not ignore:
            lines.append([X[i], X[neighs[n]]])
    i += 1

lc = mc.LineCollection(lines, linewidths=2)
fig, ax = plt.subplots()
ax.add_collection(lc)
ax.autoscale()
ax.margins(0.1)
plt.scatter(X[:, 0], X[:, 1])
plt.show()
