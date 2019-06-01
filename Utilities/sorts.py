
def mergesort(points, idx, p=None, r=None, descending=False):
    if p is None:
        p = 0
    if r is None:
        r = len(points) - 1
    if p < r:
        m = int((p + r - 1)/2)
        mergesort(points, idx, p, m, descending)
        mergesort(points, idx, m+1, r, descending)
        _merge(points, p, m, r, idx, descending)


def _merge(points, start, mid, end, idx, descending):
    left = points[start:mid+1]
    right = points[mid+1:end+1]

    k = start

    while len(left) != 0 and len(right) != 0:
        if not descending:
            if left[0, idx] <= right[0, idx]:
                points[k] = left.pop(0)
            else:
                points[k] = right.pop(0)
        else:
            if left[0, idx] >= right[0, idx]:
                points[k] = left.pop(0)
            else:
                points[k] = right.pop(0)
        k += 1

    while len(left) != 0:
        points[k] = left.pop(0)
        k += 1

    while len(right) != 0:
        points[k] = right.pop(0)
        k += 1


def quicksort(points, idx, p=None, r=None, descending=False):
    if p is None:
        p = 0
    if r is None:
        r = len(points) - 1
    if p >= r:
        return
    q = _partition(points, idx, p, r, descending)
    quicksort(points, idx, p, q, descending)
    quicksort(points, idx, q, r, descending)


def quickselect(points, rank, idx, p=None, r=None):
    if rank < 0 or rank > len(points) - 1:
        raise ValueError("Rank must be a viable index of points")
    if p is None:
        p = 0
    if r is None:
        r = len(points) - 1
    q = _partition(points, p, r, idx)
    if q < rank:
        quickselect(points, rank, idx, p, q)
    elif q > rank:
        quickselect(points, rank, idx, q, r)


def _partition(points, p, r, idx, descending=False):
    q = p
    for i in range(p, r + 1):
        if not descending:
            if points[r].coord[idx] < points[i].coord[idx]:
                _swap(points, q, i)
                q += 1
        else:
            if points[r].coord[idx] > points[i].coord[idx]:
                _swap(points, q, i)
                q += 1
    _swap(points, q, r)
    return q


def _swap(points, i, j):
    temp = points[i]
    points[i] = points[j]
    points[j] = temp
