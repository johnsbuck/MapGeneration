import numpy as np
from MeshGenerator.TriangularGrid import TriangularGrid


class TriangularGridGenerator(object):

    def __init__(self, resolution=80, extent=np.array([1, 1]), offset=np.array([0.5, 0.5])):
        self.extent = extent
        self.offset = offset
        self.resolution = resolution

    @property
    def extent(self):
        return self._extent

    @property
    def offset(self):
        return self._offset

    @property
    def resolution(self):
        return self._resolution

    @extent.setter
    def extent(self, value):
        if type(value) is not np.ndarray:
            raise ValueError("Extent must be a NumPy array.")
        if value.shape != (2,):
            raise ValueError("Extent must be a 2x1 NumPy array.")
        self._extent = value

    @offset.setter
    def offset(self, value):
        if type(value) is not np.ndarray:
            raise ValueError("Offset must be a NumPy array.")
        if value.shape != (2,):
            raise ValueError("Offset must be a 2x1 NumPy array.")
        self._offset = value

    @resolution.setter
    def resolution(self, value):
        if type(value) is not int:
            raise ValueError("Resolution must be an integer.")
        if value <= 0:
            raise ValueError("Resolution must be greater than 0.")
        self._resolution = value

    def generate_tri_grid_mesh(self):
        vertices = np.zeros(((self._resolution + 1) ** 2, 3))
        triangles = np.zeros((self._resolution * self._resolution * 2, 3)).astype(int)
        step_size_x = self._extent[0] / self._resolution
        step_size_y = self._extent[1] / self._resolution

        # Set Vertices, UV, Normals, and Colors
        v = 0
        for y in range(self._resolution + 1):
            for x in range(self._resolution + 1):
                vertices[v] = np.array([x * step_size_x - self._offset[0], y * step_size_y - self._offset[1], 0])
                v += 1

        # Set Triangles
        t = 0
        v = 0
        for y in range(self._resolution):
            for x in range(self._resolution):
                triangles[t, 0] = v
                triangles[t, 1] = v + self._resolution + 1
                triangles[t, 2] = v + 1
                triangles[t + 1, 0] = v + 1
                triangles[t + 1, 1] = v + self._resolution + 1
                triangles[t + 1, 2] = v + self._resolution + 2
                t += 2
                v += 1
            v += 1

        return TriangularGrid(vertices, triangles, self._resolution)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from PIL import Image, ImageDraw

    dimensions = [255, 255]

    gen = TriangularGridGenerator(5, np.array(dimensions), np.array([0., 0.]))
    tri = gen.generate_tri_grid_mesh()
    fig1, ax1 = plt.subplots()
    ax1.set_aspect('equal')
    tpc = ax1.tripcolor(tri.mesh.vertices[:, 0], tri.mesh.vertices[:, 1], tri.mesh.polygons,
                        np.zeros((tri.mesh.vertices.shape[0])), cmap="viridis", edgecolor="white")
    plt.show()

    '''
    im = Image.new("RGBA", tuple(dimensions))
    d = ImageDraw.Draw(im)
    for triangle in tri.mesh.polygons:
        poly = [(tri.mesh.vertices[triangle[0], 0], tri.mesh.vertices[triangle[0], 1]),
                (tri.mesh.vertices[triangle[1], 0], tri.mesh.vertices[triangle[1], 1]),
                (tri.mesh.vertices[triangle[2], 0], tri.mesh.vertices[triangle[2], 1])]
        d.polygon(poly, fill='red')
    im.show()
    '''
