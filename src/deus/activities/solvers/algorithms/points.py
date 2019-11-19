import numpy as np

__all__ = ['BayesPoint', 'XFPoint']

class BayesPoint:
    def __init__(self, x=None, log_pi=None, log_l=None, weight=None):
        self.x = x
        self.log_pi = log_pi
        self.log_l = log_l
        self.weight = weight

    def __eq__(self, other):
        return self.log_l == other.log_l

    def __lt__(self, other):
        return self.log_l < other.log_l

    @staticmethod
    def from_(xfpoints):
        if isinstance(xfpoints, list):
            return [BayesPoint(x=point.x,
                              log_pi=None, log_l=point.f, weight=None)
                    for point in xfpoints]
        else:
            return BayesPoint(x=xfpoints.x,
                              log_pi=None, log_l=xfpoints.f, weight=None)

    @classmethod
    def coords_of(cls, bpts):
        assert bpts is not None, "The list of points is none."

        n, d = len(bpts), len(bpts[0].x)
        coords = np.ndarray((n, d))
        for i, lpoint in enumerate(bpts):
            coords[i, :] = lpoint.x
        return coords


class XFPoint:
    def __init__(self, x=None, f=None):
        self.x = x
        self.f = f

    def __eq__(self, other):
        return self.f == other.f

    def __lt__(self, other):
        return self.f < other.f

    @classmethod
    def list_from(cls, coords, fvalues):
        nx, d = np.shape(coords)
        nf = len(fvalues)
        assert nx == nf, \
            "The number of points must be equal to number of fvalues."
        return [XFPoint(coords[i, :], fvalues[i]) for i in range(nx)]

    @classmethod
    def coords_of(cls, xfpts):
        assert xfpts is not None, "The list of points is none."

        n, d = len(xfpts), len(xfpts[0].x)
        coords = np.ndarray((n, d))
        for i, lpoint in enumerate(xfpts):
            coords[i, :] = lpoint.x
        return coords

    @classmethod
    def fvalues_of(cls, xfpts):
        assert xfpts is not None, "The list of points is none."

        n = len(xfpts)
        fvalues = np.ndarray(n)
        for i, lpoint in enumerate(xfpts):
            fvalues[i] = lpoint.f
        return fvalues
