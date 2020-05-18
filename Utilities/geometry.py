"""Geometry

Used for obtaining specifics on given polygons or lines
"""

from sys import float_info
from Utilities.metric import *


def centroid(points):
    """ Obtains the centroid of a 2D polygon using its vertices.

    Args:
        points (list): List of vertices for 2D polygon.

    Returns:
        (list) A 2D-Vector that is the centroid for the polygon.
    """
    return [sum(points[:, 0]) / len(points), sum(points[:, 1]) / len(points)]


def is_between(a, b, c, dist=euclidean):
    return abs(dist(a, c) + dist(c, b) - dist(a, b)) < float_info.epsilon


def line_intersection(p, q, r, s):
    """ Obtain the intersection between two lines

    Args:
        p (list): Point 1 of Line 1
        q (list): Point 2 of Line 1
        r (list): Point 1 of Line 2
        s (list): Point 2 of Line 2

    Returns:
        (list) Point of intersection if it exists. Otherwise, None.
    """
    cross = (r[0] * s[1]) - (r[1] * s[0])
    if abs(cross) < float_info.epsilon:
        return None

    vx = q.x - p.x
    vy = q.y - p.y
    t = (vx * s.y - vy * s.x) / cross
    intersect = [p.x + t * r.x, p.y + t * r.y]
    return intersect


def line_segment_intersection(a, b, c, d):
    """Checks if two lines intersect.

    Args:
        a (list): Point 1 of Line 1
        b (list): Point 2 of Line 1
        c (list): Point 1 of Line 2
        d (list): Point 2 of Line 2

    Returns:
        (bool) True if there is an intersection, otherwise False.
    """
    c1 = (d[1] - a[1]) * (c[0] - a[0]) > (c[1] - a[1]) * (d[0] - a[0])
    c2 = (d[1] - b[1]) * (c[0] - b[0]) > (c[1] - b[1]) * (d[0] - b[0])
    c3 = (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])
    c4 = (d[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (d[0] - a[0])

    return c1 != c2 and c3 != c4
