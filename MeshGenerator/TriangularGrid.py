from MeshGenerator.Mesh import Mesh


class TriangularGrid(object):
    def __init__(self, vertices, triangles, resolution):
        self.resolution = resolution
        self.mesh = Mesh(vertices, triangles)

    @property
    def mesh(self):
        return self._mesh

    @property
    def resolution(self):
        return self._resolution

    @mesh.setter
    def mesh(self, value):
        if type(value) is not Mesh:
            raise ValueError("The mesh must be a Mesh object.")
        self._mesh = value

    @resolution.setter
    def resolution(self, value):
        if type(value) is not int:
            raise ValueError("Resolution must be an integer.")
        if value <= 0:
            raise ValueError("Resolution must be greater than 0.")
        self._resolution = value
