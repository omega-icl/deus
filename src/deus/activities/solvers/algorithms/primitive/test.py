import numpy as np
import unittest

from deus.activities.solvers.algorithms.primitive.distance import ComputeDistance
from deus.activities.solvers.algorithms.primitive.sumb import SamplingUniformlyMultipleBodies
from deus.activities.solvers.algorithms.primitive.suob import SamplingUniformlyOneBox, SamplingUniformlyOneEllipsoid


class TestComputeDistance(unittest.TestCase):
    def test_taxicab(self):
        obj = ComputeDistance("taxicab")
        a = [1.0, 2.0]
        b = [2.0, 4.0]
        d_correct = 3.0
        d_ab = obj.distance(np.array([a]), np.array([b]))
        self.assertEqual(np.shape(d_ab), (1, 1))
        self.assertEqual(d_ab[0, 0], d_correct)
        d_ba = obj.distance(np.array([b]), np.array([a]))
        self.assertEqual(d_ab, d_ba)

        d = obj.distance(np.array([a, b]), np.array([a, b]))
        self.assertEqual(np.shape(d), (2, 2))
        self.assertEqual(d[0, 0], 0.0)
        self.assertEqual(d[1, 1], 0.0)
        self.assertEqual(d[0, 1], d[1, 0])
        self.assertEqual(d[0, 1], d_correct)

    def test_euclidean(self):
        obj = ComputeDistance("euclidean")
        a = [1.0, 2.0]
        b = [4.0, 6.0]
        d_correct = 5.0
        d_ab = obj.distance(np.array([a]), np.array([b]))
        self.assertEqual(np.shape(d_ab), (1, 1))
        self.assertEqual(d_ab[0, 0], d_correct)
        d_ba = obj.distance(np.array([b]), np.array([a]))
        self.assertEqual(d_ab, d_ba)

        d = obj.distance(np.array([a, b]), np.array([a, b]))
        self.assertEqual(np.shape(d), (2, 2))
        self.assertEqual(d[0, 0], 0.0)
        self.assertEqual(d[1, 1], 0.0)
        self.assertEqual(d[0, 1], d[1, 0])
        self.assertEqual(d[0, 1], d_correct)


class TestSamplingUniformlyOneBox(unittest.TestCase):
    def test_1d_box_sampling(self):
        lb, ub = 1.0, 3.0
        box = {"lcorner": np.array([lb]), "ucorner": np.array([ub])}
        alg = SamplingUniformlyOneBox(box)
        n = 100
        pts = alg.sample(n)
        self.assertEqual(np.shape(pts), (n, 1))
        self.assertTrue(((lb <= pts) & (pts <= ub)).all())

    def test_2d_box_sampling(self):
        lb, ub = 1.0, 3.0
        box = {"lcorner": np.array([lb, lb]), "ucorner": np.array([ub, ub])}
        alg = SamplingUniformlyOneBox(box)
        n = 100
        pts = alg.sample(n)
        self.assertEqual(np.shape(pts), (n, 2))
        self.assertTrue(((lb <= pts) & (pts <= ub)).all())

    def test_4d_box_sampling(self):
        lb, ub = 1.0, 3.0
        box = {"lcorner": np.array([lb]*4), "ucorner": np.array([ub]*4)}
        alg = SamplingUniformlyOneBox(box)
        n = 100
        pts = alg.sample(n)
        self.assertEqual(np.shape(pts), (n, 4))
        self.assertTrue(((lb <= pts) & (pts <= ub)).all())


class TestSamplingUniformlyOneEllipsoid(unittest.TestCase):
    def test_1d_ellipsoid_sampling(self):
        sdev = 2.0
        cov = np.array([[sdev * sdev]])
        ellipsoid = {"centre": np.array([0.0]), "cholesky": np.linalg.cholesky(cov), "scale": 1.0}
        alg = SamplingUniformlyOneEllipsoid(ellipsoid)
        n = 100
        pts = alg.sample(n)
        self.assertEqual(np.shape(pts), (n, 1))
        self.assertTrue(((-sdev <= pts) & (pts <= sdev)).all())

    def test_2d_ellipsoid_sampling(self):
        sdev = (5.0, 2.0)
        cov = np.diag([value**2 for value in sdev])
        ellipsoid = {"centre": np.zeros(2), "cholesky": np.linalg.cholesky(cov), "scale": 1.0}
        alg = SamplingUniformlyOneEllipsoid(ellipsoid)
        n = 100
        pts = alg.sample(n)
        self.assertEqual(np.shape(pts), (n, 2))
        for i in range(2):
            self.assertTrue(((-sdev[i] <= pts[:, i]) & (pts[:, i] <= sdev[i])).all())

    def test_4d_ellipsoid_sampling(self):
        sdev = np.arange(1, 5)
        cov = np.diag([value**2 for value in sdev])
        ellipsoid = {"centre": np.zeros(4), "cholesky": np.linalg.cholesky(cov), "scale": 1.0}
        alg = SamplingUniformlyOneEllipsoid(ellipsoid)
        n = 100
        pts = alg.sample(n)
        self.assertEqual(np.shape(pts), (n, 4))
        for i in range(2):
            self.assertTrue(((-sdev[i] <= pts[:, i]) & (pts[:, i] <= sdev[i])).all())


class TestSamplingUniformlyMultipleBodies(unittest.TestCase):
    def test_1d_multiple_boxes_sampling(self):
        lb1, ub1 = 1.0, 3.0
        lb2, ub2 = 1.0, 3.0

        vol1 = ub1 - lb1
        vol2 = ub2 - lb2
        bodies = [
            {"type": "box",
             "definition": {"lcorner": np.array([lb1]), "ucorner": np.array([ub1])},
             "volume": vol1},
            {"type": "box",
             "definition": {"lcorner": np.array([lb2]), "ucorner": np.array([ub2])},
             "volume": vol2}
        ]
        alg = SamplingUniformlyMultipleBodies()
        alg.set_bodies(bodies)
        n = 100
        pts = alg.sample(n)
        self.assertEqual(np.shape(pts), (n, 1))
        self.assertTrue((((lb1 <= pts) & (pts <= ub1)) ^
                         ((lb2 <= pts) & (pts <= ub2))).all())

