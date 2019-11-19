import abc
import numpy as np
import random

from deus.activities.solvers.algorithms import PrimitiveAlgorithm
from deus.activities.solvers.algorithms.primitive.suob import \
    SamplingUniformlyOneBox, SamplingUniformlyOneEllipsoid

'''
sumb - Sampling Uniformly Multiple Bodies
'''


class SamplingUniformlyMultipleBodies(PrimitiveAlgorithm):
    def __init__(self):
        super().__init__()
        self.box_algo = SamplingUniformlyOneBox()
        self.ell_algo = SamplingUniformlyOneEllipsoid()

        self.bodies = None

        self._n_dims = None

    @staticmethod
    def get_type():
        return "sumb"

    def get_ui_name(self):
        return "mix"

    def set_bodies(self, bodies):
        assert isinstance(bodies, list), \
            "bodies must be a list."
        assert all(isinstance(item, dict) for item in bodies), \
            "All items in bodies list are a dictionary"
        mkeys = ['type', 'definition', 'volume']
        for i, item in enumerate(bodies):
            assert all(key in item for key in mkeys), \
                "body " + str(i) + " does not have the proper keys, i.e. " \
                "['type', 'definition', 'volume']"
        self.bodies = bodies

    def sample(self, npts):
        assert self.bodies is not None, \
            "Bodies must be defined."
        self.find_dimensionality()

        bodies_index = np.arange(len(self.bodies)).tolist()
        selection_pr = np.asarray([body["volume"] for body in self.bodies])
        total = np.sum(selection_pr)
        selection_pr /= total

        samples = np.empty((0, self._n_dims))
        n_samples = 0
        while n_samples < npts:
            chosen_body_idx = int(np.random.choice(bodies_index,
                                                   size=1,
                                                   p=selection_pr))

            sample = self.sample_one_body_by_index(chosen_body_idx)[0]

            ne = self.how_many_bodies_contain(sample)

            u, a = float(np.random.uniform(0, 1, size=1)), 1.0/ne
            if u <= a:
                samples = np.append(samples, [sample], axis=0)
                n_samples += 1
        return samples

    def find_dimensionality(self):
        assert self.bodies is not None, \
            "Bodies must be defined."
        first_body = self.bodies[0]
        if first_body["type"] == "box":
            self._n_dims = len(first_body["definition"]["lcorner"])
        elif first_body["type"] == "ellipsoid":
            self._n_dims = len(first_body["definition"]["centre"])

    def sample_one_body_by_index(self, idx):
        body = self.bodies[idx]
        if body["type"] == "box":
            self.box_algo.set_body(body["definition"])
            x = self.box_algo.sample(1)

        elif body["type"] == "ellipsoid":
            self.ell_algo.set_body(body["definition"])
            x = self.ell_algo.sample(1)

        return x

    def how_many_bodies_contain(self, x):
        ans = 0
        for body in self.bodies:
            if body["type"] == "box":
                self.box_algo.set_body(body["definition"])
                is_contained = self.box_algo.body_contains(x)
            elif body["type"] == "ellipsoid":
                self.ell_algo.set_body(body["definition"])
                is_contained = self.ell_algo.body_contains(x)
            else:
                assert False, \
                    "The body must be in list ['box', 'ellipsoid']."
            if is_contained:
                ans += 1
        return ans
