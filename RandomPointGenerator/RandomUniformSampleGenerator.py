import numpy as np


class RandomUniformSampleGenerator(object):

    def __init__(self, num_points, extent, seed=None):
        # --------------------------------
        # User Parameters
        # --------------------------------

        # Define self._num_points
        self.num_points = num_points
        # Define both self._extent and self._dim
        self.extent = extent
        # Define self._seed
        self.seed = seed

        # --------------------------------
        # Sample List
        # --------------------------------
        self._samples = np.array().reshape((0, 2))

    def __call__(self, *args, **kwargs):
        return self.generate()

    def generate(self):
        np.random.seed(self._seed)
        self._samples = []
        for i in range(self._num_points):
            self._samples.append([np.random.uniform(0, self._extent[0]), np.random.uniform(0, self._extent[1])])

        self._samples = np.array(self._samples)
        return self._samples

    @property
    def num_points(self):
        return self._num_points

    @property
    def extent(self):
        return self._extent

    @property
    def seed(self):
        return self._seed

    @property
    def samples(self):
        if len(self._samples) == 0:
            return np.empty((0, 2))
        return self._samples
