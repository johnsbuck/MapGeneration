from sys import float_info


def centroid(points):
    return [sum(points[:, 0]) / len(points), sum(points[:, 1]) / len(points)]


def line_intersection(p, q, r, s):
    cross = (r[0] * s[1]) - (r[1] * s[0])
    if abs(cross) < float_info.epsilon:
        return None

    vx = q.x - p.x
    vy = q.y - p.y
    t = (vx * s.y - vy * s.x) / cross
    intersect = [p.x + t * r.x, p.y + t * r.y]
    return intersect


def line_segment_intersection(a, b, c, d):
    c1 = (d[1] - a[1]) * (c[0] - a[0]) > (c[1] - a[1]) * (d[0] - a[0])
    c2 = (d[1] - b[1]) * (c[0] - b[0]) > (c[1] - b[1]) * (d[0] - b[0])
    c3 = (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])
    c4 = (d[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (d[0] - a[0])

    return c1 != c2 and c3 != c4
