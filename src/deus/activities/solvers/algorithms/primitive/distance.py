import abc
import numpy as np

from deus.activities.solvers.algorithms import PrimitiveAlgorithm


'''
dist_comp - Distance Computation
'''


class ComputeDistance(PrimitiveAlgorithm):
    def __init__(self, metric=None):
        super().__init__()
        if metric is None:
            self._metric = None
        else:
            self.set_metric(metric)

    @staticmethod
    def get_type():
        return "dist_comp"

    def get_ui_name(self):
        return ""

    def set_metric(self, m):
        accepted_metrics = ['taxicab', 'euclidean']
        assert m in accepted_metrics, \
            "The metric must belong to ['taxicab', 'euclidean']."
        self._metric = m

    def distance(self, a, b):
        """
        :param a: an N x D array representing D-dimensional point(s), N > 0
        :param b: an M x D array representing D-dimensional points(s), M > 0
        :return: an array N x M representing Manhattan distance from point i to
        point j
        """
        assert self._metric is not None, \
            "Unspecified metric."
        assert isinstance(a, np.ndarray), \
            "a must be an N x D array."
        assert isinstance(b, np.ndarray), \
            "b must be an M x D array."

        shape_a, shape_b = np.shape(a), np.shape(b)
        na, nb, dims = shape_a[0], shape_b[0], 1
        if len(shape_a) > 1:
            dims = shape_a[1]
            assert shape_a[1] == shape_b[1], \
                "a and b must have the same number of columns (dimensions)."

        a_expanded = np.ndarray((na, nb * dims))
        b_expanded = np.ndarray((na, nb * dims))
        for j in range(nb):
            a_expanded[:, j * dims:(j + 1) * dims] = a
            b_expanded[:, j * dims:(j + 1) * dims] = np.array([b[j, :], ] * na)
        delta = a_expanded - b_expanded

        d = np.ndarray((na, nb))
        if self._metric == 'taxicab':
            abs_delta = np.abs(delta)
            for j in range(nb):
                d[:, j] = np.sum(abs_delta[:, j * dims:(j + 1) * dims], axis=1)
        elif self._metric == 'euclidean':
            sq_delta = delta**2
            for j in range(nb):
                d[:, j] = np.sqrt(
                    np.sum(sq_delta[:, j * dims:(j + 1) * dims], axis=1))
        return d
