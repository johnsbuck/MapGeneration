"""Simplex Noise
This class is used for generating 1D, 2D, or 3D simplex noise based on given parameters.

For more information, look into the references below.

References:
    * Patel, Amit. “Making maps with noise functions.” Making maps with noise functions, Red Blob Games, 2015,
      www.redblobgames.com/maps/terrain-from-noise/.
    * Flick, Jasper. “Noise, being a pseudorandom artist.” Noise, being a pseudorandom artist, Catlike Coding, 2014,
      catlikecoding.com/unity/tutorials/noise/.
    * Flick, Jasper. “Simplex Noise, keeping it simple.” Simplex Noise, keeping it simple, Catlike Coding, 2015,
      https://catlikecoding.com/unity/tutorials/simplex-noise/
"""

import numpy as np
from NoiseGenerator.INoise import INoise
from NoiseGenerator.NoiseSample import NoiseSample
from Utilities.math import *


class SimplexNoise(INoise):

    def __init__(self, seed=None):
        self._HASH_TABLE = [151, 160, 137, 91, 90, 15, 131, 13, 201, 95, 96, 53, 194, 233, 7, 225, 140, 36,
                            103, 30, 69, 142, 8, 99, 37, 240, 21, 10, 23, 190, 6, 148, 247, 120, 234, 75,
                            0, 26, 197, 62, 94, 252, 219, 203, 117, 35, 11, 32, 57, 177, 33, 88, 237, 149,
                            56, 87, 174, 20, 125, 136, 171, 168, 68, 175, 74, 165, 71, 134, 139, 48, 27, 166,
                            77, 146, 158, 231, 83, 111, 229, 122, 60, 211, 133, 230, 220, 105, 92, 41, 55, 46,
                            245, 40, 244, 102, 143, 54, 65, 25, 63, 161, 1, 216, 80, 73, 209, 76, 132, 187,
                            208, 89, 18, 169, 200, 196, 135, 130, 116, 188, 159, 86, 164, 100, 109, 198, 173,
                            186, 3, 64, 52, 217, 226, 250, 124, 123, 5, 202, 38, 147, 118, 126, 255, 82, 85,
                            212, 207, 206, 59, 227, 47, 16, 58, 17, 182, 189, 28, 42, 223, 183, 170, 213, 119,
                            248, 152, 2, 44, 154, 163, 70, 221, 153, 101, 155, 167, 43, 172, 9, 129, 22, 39,
                            253, 19, 98, 108, 110, 79, 113, 224, 232, 178, 185, 112, 104, 218, 246, 97, 228,
                            251, 34, 242, 193, 238, 210, 144, 12, 191, 179, 162, 241, 81, 51, 145, 235, 249,
                            14, 239, 107, 49, 192, 214, 31, 181, 199, 106, 157, 184, 84, 204, 176, 115, 121,
                            50, 45, 127, 4, 150, 254, 138, 236, 205, 93, 222, 114, 67, 29, 24, 72, 243, 141,
                            128, 195, 78, 66, 215, 61, 156, 180]

        if seed is not None and type(seed) is int:
            np.random.seed(seed)

        self._permutation = np.random.permutation(self._HASH_TABLE).tolist()
        self._HASH_MASK = 255

        self._GRADIENTS_1D = np.array([1., -1.])
        self._GRADIENTS_MASK_1D = 1

        self._GRADIENTS_2D = [np.array([1., 0.]),
                              np.array([-1., 0.]),
                              np.array([0., 1.]),
                              np.array([0., -1.]),
                              np.array([1., 1.]) / np.linalg.norm(np.array([1., 1.])),
                              np.array([-1., 1.]) / np.linalg.norm(np.array([-1., -1.])),
                              np.array([1., -1.]) / np.linalg.norm(np.array([1., -1.])),
                              np.array([-1., -1.]) / np.linalg.norm(np.array([-1., -1.]))]
        self._GRADIENTS_MASK_2D = 7

        self._SQUARES_TO_TRIANGLES = (3. - np.sqrt(3.)) / 6.
        self._TRIANGLES_TO_SQUARES = (np.sqrt(3.) - 1.) / 2

        self._NOISE_LIST = [self.noise1d, self.noise2d, self.noise3d]

    def __call__(self, resolution, method, frequency, octaves, lacunarity, persistence):
        point00 = np.array([-0.5, -0.5, 0.])
        point01 = np.array([-0.5, 0.5, 0.])
        point10 = np.array([0.5, -0.5, 0.])
        point11 = np.array([0.5, 0.5, 0.])

        if resolution < 4:
            raise ValueError("Resolution must be greater than 3")
        if type(resolution) != int:
            raise ValueError("Resolution must be an integer")

        stepSize = 1. / resolution
        data = np.zeros((resolution, resolution))
        for y in range(resolution):
            point0 = lerp((y + 0.5) * stepSize, point00, point01)
            point1 = lerp((y + 0.5) * stepSize, point10, point11)
            for x in range(resolution):
                point = lerp((x + 0.5) * stepSize, point0, point1)
                t = self.sum(method, point, frequency, octaves, lacunarity, persistence).value
                data[x, y] = self.modify_out(t)
        return data

    def _get_hash(self, val):
        return self._permutation[val % len(self._HASH_TABLE)]

    @property
    def NOISE_LIST(self):
        return self._NOISE_LIST

    @staticmethod
    def fade(t):
        return (t ** 3) * (t * (t * 6 - 15) + 10)

    @staticmethod
    def fade_derivative(t):
        return 30. * t * t * (t * (t - 2.) + 1.)

    def modify_out(self, t):
        return t * 0.5 + 0.5

    def sum(self, method, point, frequency, octaves, lacunarity, persistence):
        if frequency <= 0:
            raise ValueError("Frequency must be greater than 0")

        if octaves not in list(range(1, 17)):
            raise ValueError("Octave must be a value from 1 to 16 (inclusive)")

        if not (1 <= lacunarity <= 4):
            raise ValueError("Lacunarity must be a number from 1 to 4")

        if not (0 <= persistence <= 1):
            raise ValueError("Persistence must be a number from 0 to 1")

        sum = method(point, frequency)
        amplitude = 1.
        rng = 1.
        for o in range(1, octaves):
            frequency *= lacunarity
            amplitude *= persistence
            rng += amplitude
            sum += method(point, frequency) * amplitude
        return sum * (1. / rng)

    def noise1d(self, point, frequency):
        point *= frequency

        ix = int(np.floor(point[0]))

        sample = self._noise1d_calc(point, ix)
        sample += self._noise1d_calc(point, ix + 1)
        sample.derivative *= frequency
        return sample * 2. - 1.

    def _noise1d_calc(self, point, ix):
        x = point[0] - ix
        f = 1. - x * x
        f2 = f * f
        f3 = f * f2
        g = self._GRADIENTS_1D[self._get_hash(ix) & self._GRADIENTS_MASK_1D]
        v = g * x
        h = self._get_hash(ix)
        sample = NoiseSample(v * f3)
        sample.derivative[0] = g * f3 - 6. * h * x * f2
        return sample * 64. / 27.

    def noise2d(self, point, frequency):
        point *= frequency
        skew = (point[0] + point[1]) * self._TRIANGLES_TO_SQUARES
        sx = point[0] + skew
        sy = point[1] + skew
        ix = int(np.floor(sx))
        iy = int(np.floor(sy))
        sample = self._noise2d_calc(point, ix, iy)
        sample += self._noise2d_calc(point, ix + 1, iy + 1)
        if sx - ix >= sy - iy:
            sample += self._noise2d_calc(point, ix + 1, iy)
        else:
            sample += self._noise2d_calc(point, ix, iy + 1)
        sample.derivative *= frequency
        return sample * (8. * 2. / self._HASH_MASK) - 1.

    def _noise2d_calc(self, point, ix, iy):
        unskew = (ix + iy) * self._SQUARES_TO_TRIANGLES
        x = point[0] - ix + unskew
        y = point[1] - iy + unskew
        f = 0.5 - x * x - y * y
        sample = NoiseSample()
        if f > 0:
            f2 = f * f
            f3 = f * f2
            g = self._GRADIENTS_2D[self._get_hash(self._get_hash(ix) + iy) & self._GRADIENTS_MASK_2D]
            v = dot(g, x, y)
            v6f2 = -6. * v * f2
            sample.value = v * f3
            sample.derivative[0] = g[0] * f3 + v6f2 * x
            sample.derivative[1] = g[1] * f3 + v6f2 * y
        return sample * 2916. * float(np.sqrt(2)) / 125.

    def noise3d(self, point, frequency):
        raise NotImplementedError("Simplex Noise with 3 or more dimensions is patented under US6867776B2 " +
                                  "until 2022-Jan-08.")


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    simplex = SimplexNoise()
    resolution = 80
    step_size = 1. / resolution

    data = np.zeros((resolution, resolution))

    point00 = np.array([-0.5, -0.5, 0])
    point01 = np.array([-0.5, 0.5, 0])
    point10 = np.array([0.5, -0.5, 0])
    point11 = np.array([0.5, 0.5, 0])
    for y in range(resolution):
        point0 = lerp((y + 0.5) * step_size, point00, point01)
        point1 = lerp((y + 0.5) * step_size, point10, point11)
        for x in range(resolution):
            point = lerp((x + 0.5) * step_size, point0, point1)
            data[x, y] = simplex.modify_out(simplex.noise2d(point, 8).value)
    print(data.min(), data.max())
    plt.imshow(data, cmap="binary")
    plt.show()
