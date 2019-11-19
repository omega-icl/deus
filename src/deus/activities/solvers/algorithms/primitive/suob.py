import abc
import copy
import numpy as np
import random

from deus.activities.solvers.algorithms import PrimitiveAlgorithm


'''
suob - Sampling Uniformly One Body
'''


class SamplingUniformlyOneBody(PrimitiveAlgorithm):
    def __init__(self):
        super.__init__()

    @staticmethod
    def get_type():
        return "suob"

    @abc.abstractmethod
    def set_body(self, body):
        raise NotImplementedError

    @abc.abstractmethod
    def sample(self, npts):
        raise NotImplementedError

    @abc.abstractmethod
    def body_contains(self, point):
        raise NotImplementedError


class SamplingUniformlyOneBox(SamplingUniformlyOneBody):
    def __init__(self, box=None):
        if box is None:
            self.body = None
            self._n_dims = None
            self._ranges = None
        else:
            self.set_body(box)

    @staticmethod
    def get_ui_name():
        return "box"

    def set_body(self, body):
        assert isinstance(body, dict), \
            "the body must be a dictionary."

        lbs, ubs = body["lcorner"], body["ucorner"]
        assert isinstance(lbs, np.ndarray) and isinstance(ubs, np.ndarray), \
            "Wrong type for bounds was given. Use arrays"
        assert len(lbs) == len(ubs), \
            "Lower and upper bounds should have same size."
        assert len(lbs) > 0, \
            "Bounds cannot be zero-dimensional."

        self.body = body
        self._n_dims = len(lbs)
        self._ranges = body["ucorner"] - body["lcorner"]

    def sample(self, n_pts):
        assert self.body is not None, \
            "Define the box before sampling from it."

        samples = np.ndarray((n_pts, self._n_dims))
        u = np.asarray([[random.uniform(0.0, 1.0) for j in range(self._n_dims)]
                        for i in range(n_pts)])
        samples[:, :] = self._ranges*u + self.body["lcorner"]
        return samples

    def body_contains(self, point):
        assert self.body is not None, \
            "Define the box before checking if it contains point."
        lbs, ubs = self.body["lcorner"], self.body["ucorner"]
        ans = all(lbs[j] <= point[j] <= ubs[j]
                  for j in range(self._n_dims))
        return ans


class SamplingUniformlyOneEllipsoid(SamplingUniformlyOneBody):
    def __init__(self, ellipsoid=None):
        if ellipsoid is None:
            self.body = None
            self._n_dims = None
            self._suob_box = None
            self._inv_cholesky = None
        else:
            self.set_body(ellipsoid)

    @staticmethod
    def get_ui_name():
        return "ellipsoid"

    def set_body(self, body):
        assert isinstance(body, dict), \
            "The body must be a dictionary."

        centre, cholesky, scale = body["centre"], body["cholesky"], body["scale"]
        assert isinstance(centre, np.ndarray),\
            "The ellipsoid centre must be an array."
        assert isinstance(cholesky, np.ndarray), \
            "cholesky must be the Cholesky decomposition lower triangular matrix."
        assert isinstance(scale, float), \
            "The scale must be a float."

        self.body = body
        self._n_dims = len(centre)
        unit_box = {"lcorner": np.asarray([-0.5]*self._n_dims),
                    "ucorner": np.asarray([0.5]*self._n_dims)}
        self._suob_box = SamplingUniformlyOneBox(unit_box)
        self._inv_cholesky = np.linalg.inv(cholesky)

    def sample(self, n_pts):
        assert self.body is not None, \
            "Define the ellipsoid before sampling from it."

        # Sample from a unit box centred in the origin
        samples = self._suob_box.sample(n_pts)
        # Bring the samples on a surface of a hyper-sphere centred in the origin
        for i, p in enumerate(samples):
            s = 1 / np.sqrt(np.sum(p**2))
            samples[i, :] = s * p
        # Bring the samples inside a unit hyper-sphere centred in the origin
        power = 1.0 / self._n_dims
        for i, p in enumerate(samples):
            radius = random.uniform(0.0, 1.0)
            samples[i, :] = radius**power * p
        # Orient the samples
        samples = np.matmul(self.body["cholesky"], samples.transpose())
        samples = samples.transpose()
        # Scale the samples
        samples *= np.sqrt(self.body["scale"])
        # Translate the samples to the centre
        samples += self.body["centre"]

        return samples

    def body_contains(self, point):
        assert self.body is not None, \
            "Define the ellipsoid before checking if it contains point."
        x = copy.deepcopy(point)
        x -= self.body["centre"]
        x /= np.sqrt(self.body["scale"])
        x = np.matmul(self._inv_cholesky, x)

        norm = np.linalg.norm(x)
        ans = (norm <= 1.0)
        return ans