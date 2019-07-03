"""Value Noise
This class is used for generating 1D, 2D, or 3D value noise based on given parameters.

For more information, look into the references below.

References:
    * Patel, Amit. “Making maps with noise functions.” Making maps with noise functions, Red Blob Games, 2015,
      www.redblobgames.com/maps/terrain-from-noise/.
    * Flick, Jasper. “Noise, being a pseudorandom artist.” Noise, a Unity C# Tutorial, Catlike Coding, 2014,
      catlikecoding.com/unity/tutorials/noise/.
    * Flick, Jasper. “Noise Derivatives, Going with the Flow.” Noise Derivatives, a Unity C# Tutorial,
      Catlike Coding, 2015, catlikecoding.com/unity/tutorials/noise-derivatives/.
    * Flick, Jasper. “Simplex Noise, keeping it simple.” Simplex Noise, keeping it simple, Catlike Coding, 2015,
      https://catlikecoding.com/unity/tutorials/simplex-noise/
"""

import numpy as np
from NoiseGenerator.NoiseSample import NoiseSample
from NoiseGenerator.IHashNoise import IHashNoise
from Utilities.math import *


class ValueNoise(IHashNoise):
    """Value Noise

    A basic form of Perlin-based noise used for procedural generation.

    """
    def __init__(self, seed=None):
        """ Constructor

        Args:
            seed (int): An integer used to setting the RNG for obtaining the same value or
                will generate a new one if none is set. (Default: None)
        """

        # Hash Table used for generating Value Noise
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
                data[x, y] = t
        return data

    def _get_hash(self, val):
        """ Gives the hash of a given value.

        Args:
            val (int): An integer to be hashed.

        Returns:
            (int) The hash of the given integer.
        """
        return self._permutation[val % len(self._HASH_TABLE)]

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
            raise ValueError("Octave must be a value from 1 to 16 (inclusive)")

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
        """1-Dimensional Value Noise

        Args:
            point (np.ndarray): 1D-vector
            frequency (float): The frequency of hills and valleys within a given range.

        Returns:
            (NoiseSample) The noise generated based on the given point.
        """
        point *= frequency
        i0 = int(np.floor(point[0]))
        t = point[0] - i0

        i0 &= self._HASH_MASK
        i1 = i0 + 1

        h0 = self._get_hash(i0)
        h1 = self._get_hash(i1)

        dt = self.fade_derivative(t)
        t = self.fade(t)
        a = h0
        b = h1 - h0

        sample = NoiseSample()
        sample.value = a + b * t
        sample.derivative[0] = b * dt
        sample.derivative[1] = 0.
        sample.derivative[2] = 0.
        sample.derivative *= frequency
        return sample * (2. / self._HASH_MASK) - 1.

    def noise2d(self, point, frequency):
        """2-Dimensional Value Noise

        Args:
            point (np.ndarray): 1D-vector
            frequency (float): The frequency of hills and valleys within a given range.

        Returns:
            (NoiseSample) The noise generated based on the given point.
        """
        point *= frequency
        ix0 = int(np.floor(point[0]))
        iy0 = int(np.floor(point[1]))
        tx = point[0] - ix0
        ty = point[1] - iy0
        ix0 &= self._HASH_MASK
        iy0 &= self._HASH_MASK
        ix1 = ix0 + 1
        iy1 = iy0 + 1

        h0 = self._get_hash(ix0)
        h1 = self._get_hash(ix1)
        h00 = self._get_hash(h0 + iy0)
        h01 = self._get_hash(h0 + iy1)
        h10 = self._get_hash(h1 + iy0)
        h11 = self._get_hash(h1 + iy1)

        dtx = self.fade_derivative(tx)
        dty = self.fade_derivative(ty)
        tx = self.fade(tx)
        ty = self.fade(ty)

        a = h00
        b = h10 - h00
        c = h01 - h00
        d = h11 - h01 - h10 + h00

        sample = NoiseSample()
        sample.value = a + b * tx + (c + d * tx) * ty
        sample.derivative[0] = (b + d * ty) * dtx
        sample.derivative[1] = (c + d * tx) * dty
        sample.derivative[2] = 0.
        sample.derivative *= frequency
        return sample * (2. / self._HASH_MASK) - 1.

    def noise3d(self, point, frequency):
        """3-Dimensional Value Noise

        Args:
            point (np.ndarray): 1D-vector
            frequency (float): The frequency of hills and valleys within a given range.

        Returns:
            (NoiseSample) The noise generated based on the given point.
        """
        point *= frequency
        ix0 = int(np.floor(point[0]))
        iy0 = int(np.floor(point[1]))
        iz0 = int(np.floor(point[2]))
        tx = point[0] - ix0
        ty = point[1] - iy0
        tz = point[2] - iz0
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
        h000 = self._get_hash(h00 + iz0)
        h001 = self._get_hash(h00 + iz1)
        h010 = self._get_hash(h01 + iz0)
        h011 = self._get_hash(h01 + iz1)
        h100 = self._get_hash(h10 + iz0)
        h101 = self._get_hash(h10 + iz1)
        h110 = self._get_hash(h11 + iz0)
        h111 = self._get_hash(h11 + iz1)

        dtx = self.fade_derivative(tx)
        dty = self.fade_derivative(ty)
        dtz = self.fade_derivative(tz)
        tx = self.fade(tx)
        ty = self.fade(ty)
        tz = self.fade(tz)

        a = h00
        b = h100 - h000
        c = h010 - h000
        d = h001 - h000
        e = h110 - h010 - h100 + h000
        f = h101 - h001 - h100 + h000
        g = h011 - h001 - h010 + h000
        h = h111 - h011 - h101 + h001 - h110 + h010 + h100 - h000

        sample = NoiseSample()
        sample.value = a + b * tx + (c + e * tx) * ty + (d + f * tx + (g + h * tx) * ty) * tz
        sample.derivative[0] = (b + e * ty + (f + h * ty) * tz) * dtx
        sample.derivative[1] = (c + e * tx + (g + h * tx) * tz) * dty
        sample.derivative[2] = (d + f * tx + (g + h * tx) * ty) * dtz
        sample.derivative *= frequency
        return sample * (2. / self._HASH_MASK) - 1.


# ------------------------------------------------
# Testing Script
# ------------------------------------------------

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    v_noise = ValueNoise()
    resolution = 512
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
            data[x, y] = v_noise.sum(v_noise.noise2d, point, 4, 5, 1, 0.3).value
    plt.imshow(data, cmap="binary")
    plt.show()
