import abc
import copy
import numpy as np

from deus.activities.solvers.algorithms import PrimitiveAlgorithm
from deus.activities.solvers.algorithms.primitive.distance import ComputeDistance


class ClusteringAlgorithm(PrimitiveAlgorithm):
    def __init__(self):
        super().__init__()
        # set by client
        self._metric = None
        self._bounds = None
        # derived from settings
        self._distance_calculator = ComputeDistance()

        self._ranges = None

        # inputs
        self.points = None
        # derived from inputs
        self._n_pts = None
        self._n_dims = None

        # outputs
        self._clusters = None

    def get_type(self):
        return "clustering"

    @abc.abstractmethod
    def set_settings(self, settings):
        raise NotImplementedError

    @abc.abstractmethod
    def cluster_points(self):
        raise NotImplementedError

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

        self.points = copy.deepcopy(pts)

    def normalize_points(self):
        npts, dims = np.shape(self.points)

        lbs = self._bounds[:, 0]
        lbs_mat = np.repeat(np.array([lbs]),
                            repeats=[npts], axis=0)
        ranges_mat = np.repeat(np.array([self._ranges]),
                               repeats=[npts], axis=0)
        # normalize the points
        self.points = (self.points - lbs_mat)/ranges_mat

    def get_clusters_as_indices(self):
        return self._clusters

    def get_clusters_as_points(self):
        cap = self.clusters_as_points(self._clusters)
        return cap

    def clusters_as_points(self, clusters):
        nc = len(clusters)
        cap = []
        for i, cluster_indices in enumerate(clusters):
            cap.append(self.points[cluster_indices, :])
        return cap


class KMeans(ClusteringAlgorithm):
    def __init__(self, settings=None):
        super().__init__()
        self.__initialization = None

        self.n_clusters = None

        self.__centroids = None

        if settings is not None:
            self.set_settings(settings)

    def get_ui_name(self):
        return "kmeans"

    def set_settings(self, settings):
        self._metric = settings['metric']
        self.__initialization = settings['initialization']

        self._distance_calculator.set_metric(self._metric)

    def set_number_of_clusters(self, k):
        self.n_clusters = k

    def cluster_points(self):
        self.initialize_centroids()
        while True:
            old_centroids = copy.deepcopy(self.__centroids)
            self.update_clusters()
            self.update_centroids()
            if np.array_equal(self.__centroids, old_centroids):
                break

        '''
        1. Initialize the centroids (DONE)
        2. Compute_distance_from_points_to_centroids (DONE)
        3. Assign_points_to_centroids
        4. Update_centroids
        5. Repeat steps 2-4 until centroids stop changing
        '''

    def initialize_centroids(self):
        assert self.n_clusters is not None, \
            "Specify the number of clusters."

        if self.__initialization == "random":
            chosen = np.random.choice(np.arange(self._n_pts),
                                      self.n_clusters,
                                      replace=False)
        elif self.__initialization == "poles":
            assert self.n_clusters == 2, \
                "If INITIALIZATION is POLES then " \
                "the number of clusters must be 2."
            d = self._distance_calculator.distance(self.points,
                                                   self.points)
            chosen = np.unravel_index(d.argmax(), d.shape)
        else:
            assert False, \
                "Centroids initialization option is not recognized."
        self.__centroids = self.points[chosen, :]

    def update_clusters(self):
        distance_p2c = self._distance_calculator.distance(self.points,
                                                          self.__centroids)
        self._clusters = [[] for i in range(self.n_clusters)]
        for i in range(self._n_pts):
            parent_idx = distance_p2c[i, :].argmin()
            self._clusters[parent_idx].extend([i])

    def update_centroids(self):
        for k, children_idx in enumerate(self._clusters):
            children = self.points[children_idx, :]
            self.__centroids[k] = np.mean(children, axis=0)


class ClusteringScoreCalculator(PrimitiveAlgorithm):
    def __init__(self):
        super().__init__()
        self.__total = None

    def get_type(self):
        return "clustering"

    def get_ui_name(self):
        return "score"

    def score(self, clustering_model, criterion="bic_ellipsoid"):
        # self.find_number_of_points(clustering_model)

        log_lkhd = self.log_lkhd(clustering_model, criterion)
        complexity = self.complexity(clustering_model, criterion)
        score = 2*log_lkhd - complexity
        return score

    def find_number_of_points(self, clustering_model):
        total = 0
        for c, cluster in enumerate(clustering_model):
            size, dims = np.shape(cluster)
            total += size
        return total

    def log_lkhd(self, clustering_model, criterion="bic_ellipsoid"):
        # check if any cluster is too small
        for c, cluster in enumerate(clustering_model):
            size, dims = np.shape(cluster)
            if size <= 2 * dims:
                return -np.inf

        total = self.find_number_of_points(clustering_model)

        clusters_log_lkhd = []
        if criterion == "bic_ellipsoid":
            for c, cluster in enumerate(clustering_model):
                size, dims = np.shape(cluster)

                mean = np.mean(cluster, axis=0)
                cov = np.cov(cluster.transpose())
                inv_cov = np.linalg.inv(cov)
                det_2pi_cov = np.linalg.det(2*np.pi*cov)

                term0 = size*np.log(size/float(total))
                term1 = -size/2.0*np.log(det_2pi_cov)
                term2 = 0
                for i in range(size):
                    x = cluster[i, :]
                    delta_x = x - mean
                    delta_x_inv_cov = np.matmul(delta_x, inv_cov)
                    expon = np.matmul(delta_x_inv_cov, delta_x)
                    term2 += expon

                c_log_l = term0 + term1 - 0.5*term2
                clusters_log_lkhd.append(c_log_l)

        elif criterion == "bic_sphere":
            k = len(clustering_model)

            r_ = np.ndarray(k, dtype=int)
            ssr_ = [0]*k
            sigma = 0
            for c, cluster in enumerate(clustering_model):
                r_[c], dims = np.shape(cluster)
                centroid = np.mean(cluster, axis=0)
                centroid_mat = np.tile(centroid, (r_[c], 1))

                delta = cluster - centroid_mat
                delta_sqr = delta*delta
                ssr_[c] = np.sum(delta_sqr)
                sigma += ssr_[c]
            r = np.sum(r_)
            var_est = sigma/(r - k)

            log_2pi = np.log(2*np.pi)
            log_var = np.log(var_est)
            for c in range(k):
                term0 = - r_[c]/2.0*dims*log_2pi
                term1 = - r_[c]/2.0*dims*log_var
                term2 = - 1.0/(2.0*var_est)*ssr_[c]
                term3 = r_[c]*np.log(r_[c]/r)
                c_log_l = term0 + term1 + term2 + term3
                clusters_log_lkhd.append(c_log_l)

        log_lkhd = sum(clusters_log_lkhd)
        return log_lkhd

    def complexity(self, clustering_model, criterion="bic_ellipsoid"):
        n_points = self.find_number_of_points(clustering_model)
        k = len(clustering_model)
        size, dims = np.shape(clustering_model[0])

        if criterion == "bic_ellipsoid":
            parameters_per_cluster = 1 + dims + 0.5*dims*(dims + 1)

            n_parameters = -1
            for c, cluster in enumerate(clustering_model):
                n_parameters += parameters_per_cluster

        elif criterion == "bic_sphere":
            n_parameters = k + dims*k

        ans = n_parameters*np.log(n_points)
        return ans


class XMeansCluster:
    def __init__(self):
        self.children = None
        self.score = None
        self.splittable = True


class XMeans(ClusteringAlgorithm):
    def __init__(self, settings=None):
        super().__init__()
        self.kmeans_algo = KMeans()
        self.score_calculator = ClusteringScoreCalculator()
        self._score_criterion = None
        self._xm_clusters = []
        if settings is not None:
            self.set_settings(settings)

    def get_ui_name(self):
        return "xmeans"

    def set_settings(self, settings):
        self._metric = settings['metric']
        self._score_criterion = settings['score_criterion']
        s = {"metric": self._metric,
             "initialization": "poles"}
        self.kmeans_algo.set_settings(s)
        self.kmeans_algo.set_number_of_clusters(2)

    def set_bounds(self, bounds):
        super().set_bounds(bounds)
        self.kmeans_algo.set_bounds(self._bounds)

    def cluster_points(self):
        xm_cluster = XMeansCluster()
        xm_cluster.children = np.arange(self._n_pts).tolist()
        self._xm_clusters = [xm_cluster]

        while True:
            updated_xm_clusters = []
            for i, xcluster in enumerate(self._xm_clusters):
                if not xcluster.splittable:
                    updated_xm_clusters.append(copy.deepcopy(xcluster))
                else:
                    model1 = self.clusters_as_points([xcluster.children])
                    score1 = self.score_calculator.score(
                        model1,
                        self._score_criterion)

                    self.kmeans_algo.set_points(model1[0])
                    self.kmeans_algo.cluster_points()

                    model2 = self.kmeans_algo.get_clusters_as_points()
                    score2 = self.score_calculator.score(
                        model2,
                        self._score_criterion)

                    if score1 > score2:
                        xcluster.score = score1
                        xcluster.splittable = False
                        updated_xm_clusters.append(copy.deepcopy(xcluster))
                    else:
                        km_clusters_idx = \
                            self.kmeans_algo.get_clusters_as_indices()

                        new_cluster1 = XMeansCluster()
                        new_cluster1.children = \
                            [xcluster.children[j] for j in km_clusters_idx[0]]

                        new_cluster2 = XMeansCluster()
                        new_cluster2.children = \
                            [xcluster.children[j] for j in km_clusters_idx[1]]

                        updated_xm_clusters.extend([copy.deepcopy(new_cluster1),
                                                   copy.deepcopy(new_cluster2)])
            self._xm_clusters = updated_xm_clusters
            if any(cluster.splittable for cluster in self._xm_clusters):
                pass
            else:
                self._clusters = [cluster.children for cluster in self._xm_clusters]
                break
