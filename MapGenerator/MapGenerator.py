class MapGenerator(object):

    def __init__(self):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        return self.generate()

    def generate(self):
        raise NotImplementedError
