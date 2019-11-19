import copy
import numpy as np
import random
import scipy.stats as scistats
import time

from deus.activities.solvers.algorithms import \
    CompositeAlgorithm, \
    XFPoint
from deus.activities.solvers.algorithms.primitive.factory import \
    PrimitiveAlgorithmFactory


class NestedSamplingWithGlobalSearch(CompositeAlgorithm):
    def __init__(self, settings=None, algorithms=None):
        super().__init__()
        self._lbs = None
        self._ubs = None
        self._sorting_func = None
        self._replacement_scheme = None

        self._n_dims = None
        self._chi99_scale = None

        self.settings = None
        if settings is not None:
            self.set_settings(settings)

        self.algorithms = {
            "replacement": {
                "clustering": {"algorithm": None},
                "subclustering": {"algorithm": None},
                "sampling": {"algorithm": None}
            }
        }
        if algorithms is not None:
            self.set_algorithms(algorithms)

        self.iteration = 0
        self.live_points = None
        self.dead_points = None
        self.efficiency = None
        self.x_hat = None
        self.enlargement = None
        self.run_details = None

    @classmethod
    def get_type(cls):
        return "mc_sampling"

    @classmethod
    def get_ui_name(cls):
        return "ns_global"

    def set_settings(self, settings):
        assert settings is not None, \
            "The settings are undefined."
        assert isinstance(settings, dict), \
            "The 'settings' must be a dictionary."

        mandatory_keys = ['nlive', 'nreplacements',
                          'prng_seed', 'f0', 'alpha',
                          'stop_criteria',
                          'debug_level', 'monitor_performance']
        assert all(mkey in settings.keys() for mkey in mandatory_keys), \
            "The 'settings' keys must be the following:\n" \
            "['nlive', 'nreplacements', 'prng_seed', " \
            "'f0', 'alpha', 'stop_criteria', " \
            "'debug_level', 'monitor_performance']." \
            "Look for typos, white spaces or missing keys."

        assert isinstance(settings["stop_criteria"], list), \
            "stop_criteria must be a list of dictionaries."

        # TODO More assertions can be specified here.
        self.settings = settings

    def set_algorithms(self, algos):
        assert algos is not None, \
            "The algorithms are undefined."
        assert isinstance(algos, dict), \
            "algorithms must be a dictionary."

        replacement_steps = list(self.algorithms["replacement"].keys())
        for step in replacement_steps:
            if step in algos["replacement"]:
                algo_name = algos["replacement"][step]["algorithm"]
                if "settings" in algos["replacement"][step]:
                    algo_settings = algos["replacement"][step]["settings"]
                else:
                    algo_settings = None
                if "algorithms" in algos["replacement"][step]:
                    algo_algorithms = algos["replacement"][step]["algorithms"]
                else:
                    algo_algorithms = None
                algo = PrimitiveAlgorithmFactory.create(algo_name,
                                               algo_settings)
                self.algorithms["replacement"][step]["algorithm"] = algo

        self._replacement_scheme = self.identify_replacement_scheme()

    def set_bounds(self, lbs, ubs):
        self._lbs = lbs
        self._ubs = ubs

        self._n_dims = len(lbs)
        self._chi99_scale = scistats.chi2.ppf(0.99, self._n_dims)

    def set_sorting_func(self, f):
        self._sorting_func = f

    def get_dead_points(self):
        return copy.deepcopy(self.dead_points)

    def get_live_points(self):
        return copy.deepcopy(self.live_points)

    def empty_run_details(self):
        self.run_details = {
            "cpu_secs_for_proposals": 0.0
        }

    def run(self):
        if self.live_points is None:
            self.initialize_live_points()

        n = self.settings["nlive"]
        r = self.settings["nreplacements"]
        self.empty_run_details()

        t0 = time.time()
        proposed_points_x = self.propose_replacements(r)
        cpu_proposals = time.time() - t0
        self.run_details["cpu_secs_for_proposals"] = cpu_proposals

        proposed_points_f = self._sorting_func(proposed_points_x)

        proposed_points = XFPoint.list_from(proposed_points_x,
                                            proposed_points_f)

        self.dead_points = []
        for i, ppoint in enumerate(proposed_points):
            for j, lpoint in enumerate(self.live_points, start=0):
                if ppoint.f <= lpoint.f:
                    if j == 0:  # proposed <= current worst
                        break
                    else:  # current worst < proposed <= some live point
                        dpoint = self.live_points.pop(0)
                        self.live_points.insert(j-1, copy.deepcopy(ppoint))
                        self.dead_points.append(copy.deepcopy(dpoint))
                        break
                elif j + 1 == n:  # current best <= proposed
                    dpoint = self.live_points.pop(0)
                    self.live_points.insert(j, copy.deepcopy(ppoint))
                    self.dead_points.append(copy.deepcopy(dpoint))
                    break

        self.iteration += 1
        to_stop, c = self.is_any_stop_criterion_met()
        if to_stop:
            return "STOPPED"
        else:
            return "SUCCESS"

    def initialize_live_points(self):
        assert self._lbs is not None, "Unspecified lower bounds."
        assert self._ubs is not None, "Unspecified upper bounds."
        assert self._sorting_func is not None, \
            "Unspecified sorting function."
        seed = self.settings["prng_seed"]
        np.random.seed(seed)
        random.seed(seed)

        if self._replacement_scheme == 1:
            c_algo = self.algorithms['replacement']['clustering']['algorithm']
            bounds = np.array([self._lbs,
                               self._ubs]).transpose()
            c_algo.set_bounds(bounds)

        nlive = self.settings["nlive"]
        ubox = PrimitiveAlgorithmFactory.create("suob-box")
        box = {"lcorner": self._lbs, "ucorner": self._ubs}
        ubox.set_body(box)
        coords = ubox.sample(nlive)
        fvalues = self._sorting_func(coords)
        self.live_points = XFPoint.list_from(coords, fvalues)

        self.live_points.sort()

    def top_up_to(self, n):
        if n < self.settings['nlive']:
            assert False, "Attempt to top up to a lower number of live points."
        elif n == self.settings['nlive']:
            pass  # Nothing to do
        else:
            spawns_to_do = n - self.settings['nlive']
            while spawns_to_do > 0:
                spawned_points_x = self.propose_replacements(spawns_to_do)
                spawned_points_f = self._sorting_func(spawned_points_x)
                spawned_points = XFPoint.list_from(spawned_points_x,
                                                   spawned_points_f)

                for i, spawned in enumerate(spawned_points):
                    for j, lpoint in enumerate(self.live_points, start=0):
                        if spawned.f < lpoint.f:
                            if j == 0: #  spawned < current worst
                                break
                            else: # current worst < spawned < some live point
                                self.live_points.insert(
                                    j, copy.deepcopy(spawned))
                                self.settings['nlive'] += 1
                                break
                        # current best <= spawned
                        elif j + 1 == len(self.live_points):
                            self.live_points.insert(
                                j+1, copy.deepcopy(spawned))
                            self.settings['nlive'] += 1
                            break

                spawns_to_do = n - self.settings['nlive']

    def evaluate_live_points_fvalue(self):
        coords = XFPoint.coords_of(self.live_points)
        fvalues = self._sorting_func(coords)
        self.live_points = XFPoint.list_from(coords, fvalues)

    def identify_replacement_scheme(self):
        subclustering = self.algorithms['replacement']['subclustering']['algorithm']
        clustering = self.algorithms['replacement']['clustering']['algorithm']
        sampling = self.algorithms['replacement']['sampling']['algorithm']

        scheme = None
        if clustering is None:
            if sampling.get_type() in ["suob"]:
                scheme = 0  # global search without clustering
        else:
            if subclustering is None:
                scheme = 1  # global search with clustering only
            else:
                scheme = 2  # global search with clustering and subclustering

        assert scheme is not None, \
            "The algorithms for 'replacement' are not valid."
        return scheme

    def propose_replacements(self, r):
        clus_algo = self.algorithms['replacement']['clustering']['algorithm']
        samp_algo = self.algorithms['replacement']['sampling']['algorithm']

        # Determine the body/ies
        if self._replacement_scheme == 0:
            self.compute_enlargement()

            lpts_coords = XFPoint.coords_of(self.live_points)
            if samp_algo.get_ui_name() == "box":
                if self._n_dims == 1:
                    lc, uc = np.array([np.min(lpts_coords)]), \
                             np.array([np.max(lpts_coords)])
                else:
                    lc, uc = np.min(lpts_coords, axis=0), \
                             np.max(lpts_coords, axis=0)
                ranges = uc - lc
                f = self.enlargement
                lc -= f*ranges
                uc += f*ranges
                envelope = {"lcorner": lc, "ucorner": uc}

            elif samp_algo.get_ui_name() == "ellipsoid":
                centre = np.mean(lpts_coords, axis=0)
                cov = np.cov(lpts_coords.transpose())
                chol_mat = np.linalg.cholesky(cov)
                f = self.enlargement
                scale = (1.0 + f)*self._chi99_scale
                envelope = {"centre": centre, "cholesky": chol_mat, "scale": scale}

            else:
                assert False, "Body type not recognized."

            samp_algo.set_body(envelope)

        elif self._replacement_scheme == 1:
            self.compute_enlargement()

            lpts_coords = XFPoint.coords_of(self.live_points)

            clus_algo.set_points(lpts_coords)
            clus_algo.normalize_points()
            clus_algo.cluster_points()
            # list of Clusters As Indices
            list_of_cai = clus_algo.get_clusters_as_indices()
            # list of Clusters As Original Points
            list_of_caop = [lpts_coords[indices, :] for indices in list_of_cai]
            print('# of clusters', len(list_of_caop))  # Just for testing

            bodies = []
            n = self.settings['nlive']
            for k, caop in enumerate(list_of_caop):
                centre = np.mean(caop, axis=0)
                cov = np.cov(caop.transpose())
                chol_mat = np.linalg.cholesky(cov)

                n_k, n_dims = np.shape(caop)
                f_ik = self.enlargement*np.sqrt(float(n)/n_k)
                scale = (1.0 + f_ik)*self._chi99_scale

                envelope = {"centre": centre,
                            "cholesky": chol_mat,
                            "scale": scale}
                volume = f_ik*np.prod(chol_mat)
                body = {
                    "type": "ellipsoid",
                    "definition": envelope,
                    "volume": volume
                }
                bodies.append(body)
            samp_algo.set_bodies(bodies)

        elif self._replacement_scheme == 2:  # subclustering
            assert False, "scheme not implemented yet"

        # Sample uniformly the body/ies
        proposed_replacements = samp_algo.sample(r)
        outside = ~self.is_within_bounds(proposed_replacements)
        while True:
            nout = np.sum(outside)
            if nout == 0:
                break
            proposed_replacements[outside, :] = samp_algo.sample(nout)
            outside = ~self.is_within_bounds(proposed_replacements)

        return proposed_replacements

    def compute_enlargement(self):
        n = self.settings["nlive"]
        f0 = self.settings["f0"]
        if self.iteration == 0:
            self.x_hat = 1.0
            self.enlargement = f0
        else:
            successful_replacements = len(self.dead_points)
            self.x_hat *= (n / float(n + 1)) ** successful_replacements
            alpha = self.settings["alpha"]
            self.enlargement = f0*self.x_hat**alpha

    def is_within_bounds(self, x):
        assert isinstance(x, np.ndarray), "x must be a N x D matrix."
        n = np.shape(x)[0]
        above_lbs = x - np.array([self._lbs, ]*n)
        below_ubs = np.array([self._ubs, ]*n) - x
        within = np.concatenate((above_lbs, below_ubs), axis=1)

        answer = np.all(within >= 0, axis=1)
        return answer

    def is_any_stop_criterion_met(self):
        for c, criterion in enumerate(self.settings["stop_criteria"]):
            for k, v in criterion.items():
                if k == "max_iterations":
                    if self.iteration == v:
                        return True, c
        return False, 0

    def get_best_point(self):
        return self.live_points[-1]

