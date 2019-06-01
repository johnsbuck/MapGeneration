import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.tri as mtri
from matplotlib.colors import hsv_to_rgb, rgb_to_hsv
from NoiseGenerator.PerlinNoise import PerlinNoise
from Utilities.math import lerp


# Storage Class
class Mesh(object):
    def __init__(self, resolution=10, vertices=[], triangles=[], colors=[], normals=[], uv=[]):
        self.resolution = resolution
        self.vertices = vertices
        self.triangles = triangles
        self.colors = colors
        self.normals = normals
        self.uv = uv

    def calculate_normals(self):
        v = 0
        for y in range(self.resolution + 1):
            for x in range(self.resolution + 1):
                self.normals[v] = np.array([0., 1., self.get_y_derivative(x, y)])
                v += 1

    def get_x_derivative(self, x, y):
        row_offset = y * (self.resolution + 1)
        if y > 0:
            back = self.vertices[(y - 1) * row_offset + x][2]
            if y < self.resolution:
                forward = self.vertices[(y + 1) * row_offset + x][2]
                scale = 0.5 * self.resolution
            else:
                forward = self.vertices[y * row_offset + x][2]
                scale = self.resolution
        else:
            back = self.vertices[y * row_offset + x][2]
            forward = self.vertices[(y + 1) * row_offset + x][2]
            scale = self.resolution
        return (forward - back) * scale

    def get_y_derivative(self, x, y):
        row_offset = self.resolution + 1
        if x > 0:
            left = self.vertices[row_offset + x - 1][2]
            if x < self.resolution:
                right = self.vertices[row_offset + x + 1][2]
                scale = 0.5 * self.resolution
            else:
                right = self.vertices[row_offset + x][2]
                scale = self.resolution
        else:
            left = self.vertices[row_offset + x][2]
            right = self.vertices[row_offset + x + 1][2]
            scale = self.resolution
        return (right - left) * scale


# Noise Generator
perlin = PerlinNoise()

# User Parameters
resolution = 40
strength = 1.
frequency = 4
octaves = 4
lacunarity = 2
persistence = 0.5
amplitude = strength / frequency

offset = np.array([1, 2, 3])
q = R.from_euler('xyz', [45, 45, 0], degrees=True)
q_inv = q.inv()
q = q.as_euler("xyz")
q_inv = q_inv.as_euler("xyz")

# Cardinal Points
point00 = q * np.array([-0.5, -0.5, 0]) + offset
point01 = q * np.array([-0.5, 0.5, 0]) + offset
point10 = q * np.array([0.5, -0.5, 0]) + offset
point11 = q * np.array([0.5, 0.5, 0]) + offset

# Initialization
vertices = np.zeros(((resolution + 1) ** 2, 3))
triangles = np.zeros((resolution * resolution * 2, 3)).astype(int)
colors = np.zeros((vertices.shape[0],))
normals = np.zeros(vertices.shape)
uv = np.zeros((vertices.shape[0], 2))
stepSize = 1. / resolution

mesh = Mesh(resolution)

# Set Vertices
v = 0
for y in range(resolution + 1):
    for x in range(resolution + 1):
        vertices[v] = np.array([x * stepSize - 0.5, y * stepSize - 0.5, 0])
        uv[v] = np.array([x * stepSize, y * stepSize])
        v += 1

# Set Colors
v = 0
for y in range(resolution + 1):
    point0 = lerp(y * stepSize, point00, point01)
    point1 = lerp(y * stepSize, point10, point11)
    for x in range(resolution + 1):
        point = lerp(x * stepSize, point0, point1)
        sample = perlin.sum(perlin.perlin2d, point, frequency, octaves, lacunarity, persistence)
        sample *= amplitude
        vertices[v, 2] = sample.value * 0.5
        sample.derivative = q_inv * sample.derivative
        normals[v] = np.array([-sample.derivative[0], -sample.derivative[1], 1.])
        colors[v] = sample.value * 0.5 + 0.5
        v += 1

# Set Triangles
t = 0
v = 0
for y in range(resolution):
    for x in range(resolution):
        triangles[t, 0] = v
        triangles[t, 1] = v + resolution + 1
        triangles[t, 2] = v + 1
        triangles[t + 1, 0] = v + 1
        triangles[t + 1, 1] = v + resolution + 1
        triangles[t + 1, 2] = v + resolution + 2
        t += 2
        v += 1
    v += 1

mesh.vertices = vertices
mesh.triangles = triangles
mesh.uv = uv
mesh.normals = normals
mesh.colors = colors

fig1, ax1 = plt.subplots()
ax1.set_aspect('equal')
tpc = ax1.tripcolor(mesh.vertices[:, 0], mesh.vertices[:, 1], mesh.triangles,
                    colors, cmap="viridis")
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_zlim(0, 1)

ax.plot_trisurf(vertices[:, 0], vertices[:, 1], vertices[:, 2], triangles=mesh.triangles,
                cmap="viridis", antialiased=False, linewidth=0, edgecolor="none")
plt.show()
