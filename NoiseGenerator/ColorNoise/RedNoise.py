import numpy as np


class RedNoise(object):

    def __init__(self):
        pass

    def gaussianBlur(self, noise, blurSigma, blurSize):



    def noise2d(self, resolution=256):
        noise = np.random.random(size=(resolution, resolution))

        sigma = 1.0
        BLUR_THRESHOLD_PERCENT = 0.005
        blurSize = int(np.floor(1. + 2. * np.sqrt(-2. * sigma * sigma * np.log(BLUR_THRESHOLD_PERCENT)))) + 1


