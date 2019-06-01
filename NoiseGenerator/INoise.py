from abc import ABCMeta, abstractmethod


class INoise(metaclass=ABCMeta):

    def __init__(self):
        raise NotImplementedError("This object is an interface that has no implementation.")

    @property
    @abstractmethod
    def NOISE_LIST(self):
        raise NotImplementedError("This object is an interface that has no implementation.")

    @abstractmethod
    def noise1d(self, point, frequency):
        raise NotImplementedError("This object is an interface that has no implementation.")

    @abstractmethod
    def noise2d(self, point, frequency):
        raise NotImplementedError("This object is an interface that has no implementation.")

    @abstractmethod
    def noise3d(self, point, frequency):
        raise NotImplementedError("This object is an interface that has no implementation.")
