"""Perlin Noise
This class is used for generating 1D, 2D, or 3D perlin noise based on given parameters.

For more information, look into the references below.

References:
    * Patel, Amit. “Making Maps with Noise Functions.” Making Maps with Noise Functions, Red Blob Games, 2015,
      www.redblobgames.com/maps/terrain-from-noise/.
    * Flick, Jasper. “Noise, being a pseudorandom artist.” Noise, a Unity C# Tutorial, Catlike Coding, 2014,
      catlikecoding.com/unity/tutorials/noise/.
    * Flick, Jasper. “Noise Derivatives, Going with the Flow.” Noise Derivatives, a Unity C# Tutorial,
      Catlike Coding, 2015, catlikecoding.com/unity/tutorials/noise-derivatives/.

TODO: Add OpenCL or Vulkan Compute to speed up processing and allow quicker and larger terrain generation.
"""

import numpy as np
from NoiseGenerator.NoiseSample import NoiseSample
from NoiseGenerator.IHashNoise import IHashNoise
from Utilities.math import *


class PerlinNoise(IHashNoise):
    """Perlin Noise

    Perlin Noise is a common form of noise used for procedural generation. It is the basis for many other noises
    and is useful for the generation of various forms of terrain or natural formations.
    """

    def __init__(self, seed=None):
        """Constructor

        Args:
            seed (int): An integer used to setting the RNG for obtaining the same value or
                will generate a new one if none is set. (Default: None)
        """

        # Hash Table used for generating Perlin Noise
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

        # Set seed if one exists
        if seed is not None and type(seed) is int:
            np.random.seed(seed)

        # Generate random permutation of Hash Table for use.
        self._permutation = np.random.permutation(self._HASH_TABLE).tolist()
        self._HASH_MASK = 255

        # Set Gradients for 1D, 2D, and 3D noises
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

        self._GRADIENTS_3D = [np.array([1., 1., 0.]),
                              np.array([-1., 1., 0.]),
                              np.array([1., -1., 0.]),
                              np.array([-1., -1., 0.]),
                              np.array([1., 0., 1.]),
                              np.array([-1., 0., 1.]),
                              np.array([1., 0., -1.]),
                              np.array([-1., 0., -1.]),
                              np.array([0., 1., 1.]),
                              np.array([0., -1., 1.]),
                              np.array([0., 1., -1.]),
                              np.array([0., -1., -1.]),
                              np.array([1., 1., 0.]),
                              np.array([-1., 1., 0.]),
                              np.array([0., -1., 1.]),
                              np.array([0., -1., -1.])]
        self._GRADIENTS_MASK_3D = 15

        self._NOISE_LIST = [self.noise1d, self.noise2d, self.noise3d]

    def __call__(self, resolution, method, frequency, octaves, lacunarity, persistence):
        """Call function used to generate (resolution x resolution) points of noise with the given common parameters.

        Args:
            resolution (int): Generates (resolution x resolution) noises to be given to the user.
            method (function): A noise function (such as PerlinNoise().noise2d) that is used to generate noise.
            frequency (float): The frequency of hills and valleys within a given range.
            octaves (int): The number of noise samples applied to one point.
            lacunarity (float): The factor by which the frequency changes.
            persistence (float): The amplitude of the noise.

        Returns:
            (np.ndarray) A numpy array containing of shape (resolution, resolution).
        """

        # Coordinates of our quad that our points are generated on. Used on resolution array with linear interpolation.
        # TODO: Might add more features (user-defined bias, different point generations)
        #   Need to do more experimentation.
        point00 = np.array([-0.5, -0.5, 0])
        point01 = np.array([-0.5, 0.5, 0])
        point10 = np.array([0.5, -0.5, 0])
        point11 = np.array([0.5, 0.5, 0])

        # Type Checking
        if resolution < 4:
            raise ValueError("Resolution must be greater than 3")
        if type(resolution) != int:
            raise ValueError("Resolution must be an integer")

        # Generating points for (resolution, resolution) array.
        stepSize = 1. / resolution
        data = np.zeros((resolution, resolution))
        for y in range(resolution):
            point0 = lerp((y + 0.5) * stepSize, point00, point01)
            point1 = lerp((y + 0.5) * stepSize, point10, point11)
            for x in range(resolution):
                point = lerp((x + 0.5) * stepSize, point0, point1)
                t = self.sum(method, point, frequency, octaves, lacunarity, persistence).value
                data[x, y] = t * 0.5 + 0.5
        return data

    def _get_hash(self, val):
        """ Gives the hash of a given value.

        Args:
            val (int): An integer to be hashed.

        Returns:
            (int) The hash of the given integer.
        """
        return self._permutation[val & self._HASH_MASK]

    @property
    def NOISE_LIST(self):
        """The list of possible noise functions given within the PerlinNoise class.

        Returns:
            (list) The list of possible noise functions.
        """
        return self._NOISE_LIST

    @staticmethod
    def fade(t):
        """The smoothing function used to make the noise less sharp.

        In this case, our smoothing function is a 5th degree polynomial:
        (6t^5 - 15t^4 + 10t^3)

        Args:
            t (float): A value given to smooth.

        Returns:
            (float) The smoothed value
        """
        return (t ** 3) * (t * (t * 6 - 15) + 10)

    @staticmethod
    def fade_derivative(t):
        """The first derivative of the smoothing function.

        The smoothing function derivative is the following:
        30t^4 - 60t^3 + 30t^2

        Args:
            t (float): A value given to receive the derivative of the smoothing function.

        Returns:
            The returning derivative value.
        """
        return 30. * t * t * (t * (t - 2.) + 1.)

    def noise_mod(self, t):
        """A common modifier function for noise.

        Args:
            t (float): A noise value

        Returns:
            (float) An adjusted noise value.
        """
        return t * 0.5 + 0.5

    def sum(self, method, point, frequency, octaves, lacunarity, persistence):
        """A function which sums several noise generations to receive adjusted noise.

        Args:
            method (function): A noise function (such as PerlinNoise().noise2d) that is used to generate noise.
            point (np.ndarray): A 3D-vector to receive noise based from.
            frequency (float): The frequency of hills and valleys within a given range.
            octaves (int): The number of noise samples applied to one point.
            lacunarity (float): The factor by which the frequency changes.
            persistence (float): The amplitude of the noise.

        Returns:
            (NoiseSample) The noise for the given point based on the given parameters.
        """

        # Type Checking
        # (Based mostly on tutorial and gives an idea of good figures. May remove in the future or change as guidelines)
        if frequency <= 0:
            raise ValueError("Frequency must be greater than 0")

        if octaves not in list(range(1, 17)):
            raise ValueError("Octave must be an integer from 1 to 16 (inclusive)")

        if lacunarity <= 0:
            raise ValueError("Lacunarity must be greater than 0")

        if not (0 <= persistence <= 1):
            raise ValueError("Persistence must be a number from 0 to 1")

        # Begin Summation
        sum = NoiseSample(0.)
        amplitude = 1.
        rng = 0.
        for o in range(octaves):
            sum += method(point, frequency) * amplitude
            rng += amplitude
            frequency *= lacunarity
            amplitude *= persistence
        return sum * (1. / rng)

    def noise1d(self, point, frequency):
        """1-Dimensional Perlin Noise

        Args:
            point (np.ndarray): 1D-vector
            frequency (float): The frequency of hills and valleys within a given range.

        Returns:
            (NoiseSample) The noise generated based on the given point.
        """
        point *= frequency
        i0 = int(np.floor(point[0]))
        t0 = point[0] - i0
        t1 = t0 - 1.
        i0 &= self._HASH_MASK
        i1 = i0 + 1

        g0 = self._GRADIENTS_1D[self._get_hash(i0) & self._GRADIENTS_MASK_1D]
        g1 = self._GRADIENTS_1D[self._get_hash(i1) & self._GRADIENTS_MASK_1D]

        v0 = g0 * t0
        v1 = g1 * t1

        dt = self.fade_derivative(t0)
        t = self.fade(t0)

        a = v0
        b = v1 - v0

        da = g0
        db = g1 - g0

        sample = NoiseSample()
        sample.value = a + b * t
        sample.derivative[0] = da + db * t + b * dt
        sample.derivative[1] = 0.
        sample.derivative[2] = 0.
        sample.derivative *= frequency
        return sample * 2.

    def noise2d(self, point, frequency):
        """2-Dimensional Perlin Noise

        Args:
            point (np.ndarray): 2D-vector
            frequency (float): The frequency of hills and valleys within a given range.

        Returns:
            (NoiseSample) The noise generated based on the given point.
        """
        point *= frequency
        ix0 = int(np.floor(point[0]))
        iy0 = int(np.floor(point[1]))
        tx0 = point[0] - ix0
        ty0 = point[1] - iy0
        tx1 = tx0 - 1.
        ty1 = ty0 - 1.
        ix0 &= self._HASH_MASK
        iy0 &= self._HASH_MASK
        ix1 = ix0 + 1
        iy1 = iy0 + 1

        h0 = self._get_hash(ix0)
        h1 = self._get_hash(ix1)
        g00 = self._GRADIENTS_2D[self._get_hash(h0 + iy0) & self._GRADIENTS_MASK_2D]
        g01 = self._GRADIENTS_2D[self._get_hash(h0 + iy1) & self._GRADIENTS_MASK_2D]
        g10 = self._GRADIENTS_2D[self._get_hash(h1 + iy0) & self._GRADIENTS_MASK_2D]
        g11 = self._GRADIENTS_2D[self._get_hash(h1 + iy1) & self._GRADIENTS_MASK_2D]

        v00 = dot(g00, tx0, ty0)
        v01 = dot(g01, tx0, ty1)
        v10 = dot(g10, tx1, ty0)
        v11 = dot(g11, tx1, ty1)

        dtx = self.fade_derivative(tx0)
        dty = self.fade_derivative(ty0)
        tx = self.fade(tx0)
        ty = self.fade(ty0)

        a = v00
        b = v10 - v00
        c = v01 - v00
        d = v11 - v01 - v10 + v00

        da = g00
        db = g10 - g00
        dc = g01 - g00
        dd = g11 - g01 - g10 + g00

        sample = NoiseSample()
        sample.value = a + b * tx + (c + d * tx) * ty
        sample.derivative = da + db * tx + (dc + dd * tx) * ty
        sample.derivative[0] += (b + d * ty) * dtx
        sample.derivative[1] += (c + d * tx) * dty
        sample.derivative = np.array(sample.derivative.tolist() + [0.])
        sample.derivative *= frequency
        return sample * float(np.sqrt(2.))

    def noise3d(self, point, frequency):
        """3-Dimensional Perlin Noise

        Args:
            point (np.ndarray): 3D-vector
            frequency (float): The frequency of hills and valleys within a given range.

        Returns:
            (NoiseSample) The noise generated based on the given point.
        """
        point *= frequency
        ix0 = int(np.floor(point[0]))
        iy0 = int(np.floor(point[1]))
        iz0 = int(np.floor(point[2]))
        tx0 = point[0] - ix0
        ty0 = point[1] - iy0
        tz0 = point[2] - iz0
        tx1 = tx0 - 1.
        ty1 = ty0 - 1.
        tz1 = tz0 - 1.
        ix0 &= self._HASH_MASK
        iy0 &= self._HASH_MASK
        iz0 &= self._HASH_MASK
        ix1 = ix0 + 1
        iy1 = iy0 + 1
        iz1 = iz0 + 1

        h0 = self._get_hash(ix0)
        h1 = self._get_hash(ix1)
        h00 = self._get_hash(h0 + iy0)
        h01 = self._get_hash(h0 + iy1)
        h10 = self._get_hash(h1 + iy0)
        h11 = self._get_hash(h1 + iy1)
        g000 = self._GRADIENTS_3D[self._get_hash(h00 + iz0) & self._GRADIENTS_MASK_3D]
        g001 = self._GRADIENTS_3D[self._get_hash(h00 + iz1) & self._GRADIENTS_MASK_3D]
        g010 = self._GRADIENTS_3D[self._get_hash(h01 + iz0) & self._GRADIENTS_MASK_3D]
        g011 = self._GRADIENTS_3D[self._get_hash(h01 + iz1) & self._GRADIENTS_MASK_3D]
        g100 = self._GRADIENTS_3D[self._get_hash(h10 + iz0) & self._GRADIENTS_MASK_3D]
        g101 = self._GRADIENTS_3D[self._get_hash(h10 + iz1) & self._GRADIENTS_MASK_3D]
        g110 = self._GRADIENTS_3D[self._get_hash(h11 + iz0) & self._GRADIENTS_MASK_3D]
        g111 = self._GRADIENTS_3D[self._get_hash(h11 + iz1) & self._GRADIENTS_MASK_3D]

        v000 = dot(g000, tx0, ty0, tz0)
        v001 = dot(g001, tx0, ty0, tz1)
        v010 = dot(g010, tx0, ty1, tz0)
        v011 = dot(g011, tx0, ty1, tz1)
        v100 = dot(g100, tx1, ty0, tz0)
        v101 = dot(g101, tx1, ty0, tz1)
        v110 = dot(g110, tx1, ty1, tz0)
        v111 = dot(g111, tx1, ty1, tz1)

        dtx = self.fade_derivative(tx0)
        dty = self.fade_derivative(ty0)
        dtz = self.fade_derivative(tz0)
        tx = self.fade(tx0)
        ty = self.fade(ty0)
        tz = self.fade(tz0)

        a = v000
        b = v100 - v000
        c = v010 - v000
        d = v001 - v000
        e = v110 - v010 - v100 + v000
        f = v101 - v001 - v100 + v000
        g = v011 - v001 - v010 + v000
        h = v111 - v011 - v101 + v001 - v110 + v010 + v100 - v000

        da = g000
        db = g100 - g000
        dc = g010 - g000
        dd = g001 - g000
        de = g110 - g010 - g100 + g000
        df = g101 - g001 - g100 + g000
        dg = g011 - g001 - g010 + g000
        dh = g111 - g011 - g101 + g001 - g110 + g010 + g100 - g000

        sample = NoiseSample()
        sample.value = a + b * tx + (c + e * tx) * ty + (d + f * tx + (g + h * tx) * ty) * tz
        sample.derivative = da + db * tx + (dc + de * tx) * ty + (dd + df * tx + (dg + dh * tx) * ty) * tz
        sample.derivative[0] += (b + e * ty + (f + h * ty) * tz) * dtx
        sample.derivative[1] += (c + e * tx + (g + h * tx) * tz) * dty
        sample.derivative[2] += (d + f * tx + (g + h * tx) * ty) * dtz
        sample.derivative *= frequency
        return sample


# ------------------------------------------------
# Test Script
# ------------------------------------------------

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np

    perlin = PerlinNoise()

    point00 = np.array([-0.5, -0.5])
    point01 = np.array([-0.5, 0.5])
    point10 = np.array([0.5, -0.5])
    point11 = np.array([0.5, 0.5])

    resolution = 512
    stepSize = 1. / resolution
    data = np.zeros((resolution, resolution))
    for y in range(resolution):
        point0 = lerp((y + 0.5) * stepSize, point00, point01)
        point1 = lerp((y + 0.5) * stepSize, point10, point11)
        for x in range(resolution):
            point = lerp((x + 0.5) * stepSize, point0, point1)
            t = perlin.sum(perlin.noise2d, point, 2, 5, 1., 0.5).value
            data[x, y] = t * 0.5 + 0.5
    print(data.min(), data.max(), data.mean())

    plt.imshow(data, cmap="gray")
    plt.show()
