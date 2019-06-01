import numpy as np


class NoiseSample(object):

    def __init__(self, value=0, derivative=np.array([0., 0., 0.])):
        self.value = value
        self.derivative = derivative

    def __add__(self, other):
        if type(other) is NoiseSample:
            return NoiseSample(self.value + other.value, self.derivative + other.derivative)
        elif type(other) in (int, float):
            return NoiseSample(self.value + other, self.derivative + other)
        else:
            raise ValueError("Invalid type used for NoiseSample arithmetic operation.")

    def __iadd__(self, other):
        if type(other) is NoiseSample:
            self.value += other.value
            self.derivative += other.derivative
        elif type(other) in (int, float):
            self.value /= other
            self.derivative /= other
        else:
            raise ValueError("Invalid type used for NoiseSample arithmetic operation.")
        return self

    def __sub__(self, other):
        if type(other) is NoiseSample:
            return NoiseSample(self.value - other.value, self.derivative - other.derivative)
        elif type(other) in (int, float):
            return NoiseSample(self.value - other, self.derivative - other)
        else:
            raise ValueError("Invalid type used for NoiseSample arithmetic operation.")

    def __isub__(self, other):
        if type(other) is NoiseSample:
            self.value -= other.value
            self.derivative -= other.derivative
        elif type(other) in (int, float):
            self.value /= other
            self.derivative /= other
        else:
            raise ValueError("Invalid type used for NoiseSample arithmetic operation.")
        return self

    def __mul__(self, other):
        if type(other) is NoiseSample:
            return NoiseSample(self.value * other.value, self.derivative * other.derivative)
        elif type(other) in (int, float):
            return NoiseSample(self.value * other, self.derivative * other)
        else:
            raise ValueError("Invalid type used for NoiseSample arithmetic operation.")

    def __imul__(self, other):
        if type(other) is NoiseSample:
            self.value *= other.value
            self.derivative *= other.derivative
        elif type(other) in (int, float):
            self.value *= other
            self.derivative *= other
        else:
            raise ValueError("Invalid type used for NoiseSample arithmetic operation.")
        return self

    def __truediv__(self, other):
        if type(other) is NoiseSample:
            return NoiseSample(self.value / other.value, self.derivative / other.derivative)
        elif type(other) in (int, float):
            return NoiseSample(self.value / other, self.derivative / other)
        else:
            raise ValueError("Invalid type used for NoiseSample arithmetic operation.")

    def __truediv__(self, other):
        if type(other) is NoiseSample:
            self.value /= other.value
            self.derivative /= other.derivative
        elif type(other) in (int, float):
            self.value /= other
            self.derivative /= other
        else:
            raise ValueError("Invalid type used for NoiseSample arithmetic operation.")
        return self
