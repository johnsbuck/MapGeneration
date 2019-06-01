def lerp(t, a, b):
    return a + t * (b - a)


def dot(g, x, y, z=None):
    if z is None:
        return g[0] * x + g[1] * y
    return g[0] * x + g[1] * y + g[2] * z
