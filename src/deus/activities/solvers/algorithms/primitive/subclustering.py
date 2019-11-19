import abc
import copy
import numpy as np

from deus.activities.solvers.algorithms import PrimitiveAlgorithm
from deus.activities.solvers.algorithms.primitive.distance import ComputeDistance


class SubClusteringAlgorithm(PrimitiveAlgorithm):
    def __init__(self):
        super().__init__()
        # set by client
        self._metric = None
        self._initialization = None
        self._bounds = None

        # derived from settings
        self._distance_calculator = ComputeDistance()
        self._ranges = None

        # inputs
        self.points = None
        self.clusters = None

        # derived from inputs
        self._n_pts = None
        self._n_dims = None

        # outputs
        self._subclusters = None

    def get_type(self):
        return "subclustering"

    @abc.abstractmethod
    def set_settings(self, settings):
        raise NotImplementedError

    @abc.abstractmethod
    def execute_split_step(self):
        raise NotImplementedError

    @abc.abstractmethod
    def execute_share_step(self):
        raise NotImplementedError

    def do_subclustering(self):
        self.execute_split_step()
        self.execute_share_step()

    def set_bounds(self, bounds):
        assert isinstance(bounds, np.ndarray),\
            "bounds must be a numpy array."
        nrows, ncols = np.shape(bounds)
        assert ncols == 2,\
            "bounds must be a Nx2 numpy array."
        assert all(bounds[i, 0] < bounds[i, 1] for i in range(nrows)), \
            "lower bounds must be < upper bounds."
        self._bounds = bounds
        self._ranges = bounds[:, 1] - bounds[:, 0]
        self._n_dims = nrows

    def set_points(self, pts):
        assert isinstance(pts, np.ndarray),\
            "The points must be an N x D array."

        pts_shape = np.shape(pts)
        assert 0 < len(pts_shape), \
            "The points' shape length must be > 0."
        assert len(pts_shape) < 3, \
            "The points' shape length must be < 3."

        if len(pts_shape) == 1:
            self._n_pts, self._n_dims = pts_shape[0], 1
        else:
            self._n_pts, self._n_dims = pts_shape

        self.points = pts

    def set_clusters(self, clusters):
        assert isinstance(clusters, list),\
            "The clusters must be a list of integers."

        assert len(clusters) > 0, \
            "There has to be at least one cluster."

        for cluster in clusters:
            assert len(cluster) >= self._n_dims + 1, \
                "Each cluster must contain at least D+1 points."

        self.clusters = clusters

    def get_subclusters_as_indices(self):
        return self._subclusters

    def get_subclusters_as_points(self):
        cap = self.clusters_as_points(self._subclusters)
        return cap

    def clusters_as_points(self, clusters):
        nc = len(clusters)
        cap = []
        for i, cluster_indices in enumerate(clusters):
            cap.append(self.points[cluster_indices, :])
        return cap


class BasicSubClustering(SubClusteringAlgorithm):
    def __init__(self, settings=None):
        super().__init__()
        self.kmeans_algo = KMeans()
        if settings is not None:
            self.set_settings(settings)

    def get_ui_name(self):
        return "basic"

    def set_settings(self, settings):
        self._metric = settings['metric']
        self._initialization = settings['initialization']
        s = {"metric": self._metric,
             "initialization": self._initialization}
        self.kmeans_algo.set_settings(s)
        self.kmeans_algo.set_number_of_clusters(2)

    def execute_split_step(self):
        subclusters = self.clusters
        while True:
           '''
           1. Find richest cluster, c_j.
           2. c_j has at least 2(D+1) points?
           Yes --> Continue
           No --> STOP
           3. Use k_means to split c_j.
           4. The new cluster c_i has at least D+1 points?
           Yes --> Add it at the end of the subclusters
           No --> Assign points from c_j to c_i until c_i has D+1 points
           5. The robbed c_j has at least D+1 points?
           Yes --> Continue
           No --> Assign points from c_i to c_j until c_j has D+1 points
           '''

           n_children = [len(c) for c in subclusters]



    def execute_share_step(self):
        pass  # TODO