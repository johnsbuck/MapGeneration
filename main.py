import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d
from RandomPointGenerator import PoissonDiscSampleGenerator


poisson = PoissonDiscSampleGenerator()
points = poisson()
vor = Voronoi(points)

voronoi_plot_2d(vor)
plt.show()

'''
from RandomPointGenerator.PoissonDiscSampleGenerator import PoissonDiscSampleGenerator
from Random.RandomOrg import RandomOrg
import numpy as np
from scipy import spatial as sp
import matplotlib.pyplot as plt
from threading import Thread
from NoiseGenerator.SimplexNoise import SimplexNoise
from NoiseGenerator.PerlinNoise import PerlinNoise


random = RandomOrg("bbc790bb-7017-4bde-b415-5d860fbadff2")
poisson = PoissonDiscSampleGenerator(radius=0.5)
sample = poisson.generate()
tri = sp.Delaunay(sample)

colors = np.zeros((tri.simplices.shape[0]))

simplex = SimplexNoise()
perlin = PerlinNoise()

i = 0
for t in tri.simplices:
    centroid = np.mean(tri.points[t], axis=1)
    # colors[i] = simplex.modify_out(simplex.sum(simplex.simplex2d, centroid, 4, 6, 2, 0.5).value)
    colors[i] = perlin.simplex2d(centroid, 8)
    i += 1

fig, ax = plt.subplots()
ax.set_aspect("equal")
tpc = ax.tripcolor(tri.points[:, 0], tri.points[:, 1], tri.simplices,
                   colors, cmap="binary")
plt.show()
'''