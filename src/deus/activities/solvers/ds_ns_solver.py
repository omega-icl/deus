import copy
import numpy as np
import time

from deus.activities.solvers.validators import ValidatorForDSSolver
from deus.activities.interfaces import Subject, Observer
from deus.activities.solvers.algorithms.composite.factory import \
    CompositeAlgorithmFactory
from deus.activities.solvers.algorithms.composite import \
    NestedSamplingWithGlobalSearch
from deus.activities.solvers.evaluators.dsscores.evaluator import \
    DSScoreEvaluator
from deus.activities.solvers.evaluators.dsefp.evaluator import \
    EFPEvaluator


class DesignSpaceSolverUsingNS(Subject):
    def __init__(self, cs_path, problem=None, settings=None, algorithms=None):
        self.cs_path = cs_path

        self.problem = None
        if problem is not None:
            self.set_problem(problem)

        self.settings = None
        if settings is not None:
            self.set_settings(settings)

        self.algorithms = None
        if algorithms is not None:
            self.set_algorithms(algorithms)

        self._score_evaluator = None
        self.create_score_evaluator()

        self._efp_evaluator = None
        self.create_efp_evaluator()

        self.p_coords = None
        self.p_weights = None
        self._p_num = None
        self._p_dims = None
        self.set_format_of_coords_and_weigths_of_parameters_samples()

        self.observers = []

        self.solver_iteration = 0
        # performance data
        self.n_proposals = 0
        self.n_model_evals = 0
        self.cpu_secs_for_iteration = 0.0
        # constraints data
        self.g_info = []

        self.output_buffer = []

        # state
        self.status = "NOT_FINISHED"
        self.phase = "INITIAL"

        self.g_dim = None

        self.worst_phi = None
        self.beta, self.n_live, self.n_proposals = None, None, None

        self.main_info = None
        self.topup_info = None
        self.topup_todo = False

        self.frac_live_in_nominal_ds = None
        self.frac_live_inside_target_pds = None

    # Construction
    def set_problem(self, problem):
        # Check is already done by the activity manager.
        self.problem = problem

    def set_settings(self, settings):
        ValidatorForDSSolver.check_settings(settings)
        self.settings = settings

    def set_algorithms(self, algos):
        assert self.problem is not None, \
            "Define 'problem' before the algorithms of 'ds_solver'."
        assert self.settings is not None, \
            "Define 'settings' before the algorithms of 'ds_solver'."
        ValidatorForDSSolver.check_algorithms(algos)
        algos_permitted = [NestedSamplingWithGlobalSearch.get_type()
                           + "-" +
                           NestedSamplingWithGlobalSearch.get_ui_name()]

        sampling_algo_name = algos["sampling"]["algorithm"]
        assert sampling_algo_name in algos_permitted, \
            "The algorithm specified for 'sampling' is not permitted."

        sampling_algo_name = algos["sampling"]["algorithm"]
        sampling_algo_settings = algos["sampling"]["settings"]
        algos_for_sampling_algo = algos["sampling"]["algorithms"]
        algo = CompositeAlgorithmFactory.create(sampling_algo_name,
                                                sampling_algo_settings,
                                                algos_for_sampling_algo)
        self.algorithms = {
            "sampling": {"algorithm": algo}
        }

    def create_score_evaluator(self):
        assert self.problem is not None, \
            "Problem must be defined before creating the score evaluator."
        assert self.settings is not None, \
            "Settings must be defined before creating the score evaluator."

        score_eval = self.settings["score_evaluation"]
        evaluator_info = {
            'ufunc_script_path': self.cs_path,
            'ufunc_script_name': self.problem["user_script_filename"],
            'ufunc_name': self.problem["constraints_func_name"],
            'score_type': score_eval["score_type"]
        }
        eval_method = score_eval["method"]
        if eval_method == "serial":
            eval_options = {
                'ufunc_ptr': score_eval["constraints_func_ptr"],
                'store_constraints': score_eval["store_constraints"]
            }

        elif eval_method == "mppool":
            eval_options = {
                'pool_size': score_eval["pool_size"],
                'store_constraints': score_eval["store_constraints"]
            }

        elif eval_method == "mpi":
            assert False, "Not implemented yet."

        p_best = self.problem["parameters_best_estimate"]
        self._score_evaluator = DSScoreEvaluator(evaluator_info,
                                                 eval_method,
                                                 eval_options,
                                                 p_best)

    def create_efp_evaluator(self):
        assert self.problem is not None, \
            "Problem must be defined before creating the efp evaluator."
        assert self.settings is not None, \
            "Settings must be defined before creating the efp evaluator."

        efp_eval = self.settings["efp_evaluation"]
        evaluator_info = {
            'ufunc_script_path': self.cs_path,
            'ufunc_script_name': self.problem["user_script_filename"],
            'ufunc_name': self.problem["constraints_func_name"],
        }
        eval_method = efp_eval["method"]
        if eval_method == "serial":
            eval_options = {
                'ufunc_ptr': efp_eval["constraints_func_ptr"],
                'store_constraints': efp_eval["store_constraints"]
            }

        elif eval_method == "mppool":
            eval_options = {
                'pool_size': efp_eval["pool_size"],
                'store_constraints': efp_eval["store_constraints"]
            }

        elif eval_method == "mpi":
            assert False, "Not implemented yet."

        p_samples = self.problem["parameters_samples"]
        self._efp_evaluator = EFPEvaluator(evaluator_info,
                                           eval_method,
                                           eval_options,
                                           p_samples)

    # The most important procedure :)
    def solve(self):
        if self.phase == "INITIAL":
            self.tell_sampling_algo_the_bounds_and_sorting_function()
            self.obtain_beta_nlive_nproposals()
            self.tell_sampling_algo_the_nlive_and_nproposals()

            salgo = self.algorithms['sampling']['algorithm']
            self.reset_main_info()
            salgo.initialize_live_points()

            self.obtain_worst_phi()
            self.print_progress_summary()

            self.collect_output()
            self.notify_observers()

            print("Phase INITIAL is over.")
            self.phase = "DETERMINISTIC"

        while self.phase == "DETERMINISTIC":
            to_skip = self.settings["phases_setup"]["deterministic"]["skip"]
            if to_skip is True:
                print("Phase DETERMINISTIC is skipped.")
                self.status = "NOT_FINISHED"
                self.phase = "TRANSITION"
            else:
                # We assume that live points are sorted ascending by phi!
                t0 = time.time()
                self.reset_main_info()
                self.do_one_sampling_round()
                self.cpu_secs_for_iteration = time.time() - t0

                self.compute_frac_live_inside_nominal_ds()
                self.obtain_worst_phi()
                self.print_progress_summary()
                if self.frac_live_in_nominal_ds == 1.0:
                    print("Phase DETERMINISTIC is over.")
                    self.status = "FINISHED_DETERMINISTIC_PHASE"
                    spec = self.settings["phases_setup"]
                    if spec["nmvp_search"]["skip"] is True:
                        print("Phase NMVP_SEARCH is skipped.")
                        if spec["probabilistic"]["skip"] is True:
                            print("Phase PROBABILISTIC is skipped.")
                            self.status = "FINISHED"
                            self.collect_output()
                            self.notify_observers()
                            self.phase = "FINAL"
                        else:
                            self.collect_output()
                            self.notify_observers()
                            self.status = "NOT_FINISHED"
                            self.phase = "TRANSITION"
                    else:
                        self.collect_output()
                        self.notify_observers()
                        self.status = "NOT_FINISHED"
                        self.phase = "NMVP_SEARCH"

                else:
                    self.status = "NOT_FINISHED"
                    self.collect_output()
                    self.notify_observers()
                    self.phase = "DETERMINISTIC"

        if self.phase == "NMVP_SEARCH":
            assert False, "Not implemented yet."

        if self.phase == "TRANSITION":
            salgo = self.algorithms['sampling']['algorithm']
            self.reset_main_info()
            salgo.request_live_points_reevaluation()
            salgo.live_points.sort()

            self.obtain_worst_phi()
            self.print_progress_summary()

            self.collect_output()
            self.notify_observers()

            print("Phase TRANSITION is over.")
            self.phase = "PROBABILISTIC"

        while self.phase == "PROBABILISTIC":
            t0 = time.time()
            self.obtain_worst_phi()
            n_live_before = self.n_live
            self.obtain_beta_nlive_nproposals()
            self.reset_topup_info()
            if n_live_before == self.n_live:
                self.topup_todo = False
            else:
                self.topup_todo = True
                salgo = self.algorithms['sampling']['algorithm']
                salgo.top_up_to(self.n_live, self.n_proposals)
                self.topup_todo = False

                salgo.settings["nproposals"] = self.n_proposals

            self.reset_main_info()
            self.do_one_sampling_round()
            self.cpu_secs_for_iteration = time.time() - t0

            self.compute_frac_live_inside_target_pds()
            self.obtain_worst_phi()
            self.print_progress_summary()
            if self.frac_live_inside_target_pds == 1.0:
                print("Phase PROBABILISTIC is over.")
                self.status = "FINISHED"
                self.collect_output()
                self.notify_observers()
                self.phase = "FINAL"
            else:
                self.status = "NOT_FINISHED"
                self.collect_output()
                self.notify_observers()
                self.phase = "PROBABILISTIC"

        if self.phase == "FINAL":
            print("Design Space Characterization is done.")

    # Procedures used only during INITIAL phase
    def set_format_of_coords_and_weigths_of_parameters_samples(self):
        self.sort_parameters_samples(sort_key='w', ascending=False)

        p_samples = self.problem['parameters_samples']
        p_num = len(p_samples)
        p_dims = len(p_samples[0]['c'])

        p_coords = np.ndarray((p_num, p_dims))
        p_weights = np.ndarray(p_num)
        for i, sample in enumerate(p_samples):
            p_coords[i, :] = sample['c']
            p_weights[i] = sample['w']

        self.p_coords = p_coords
        self.p_weights = p_weights

        self._p_num, self._p_dims = np.shape(self.p_coords)

    def tell_sampling_algo_the_bounds_and_sorting_function(self):
        lbs, ubs = self.get_design_vars_bounds(which="both",
                                               as_array=True)
        sampling_algo = self.algorithms["sampling"]["algorithm"]
        sampling_algo.set_bounds(lbs, ubs)
        sampling_algo.set_sorting_func(self.phi)

    def tell_sampling_algo_the_nlive_and_nproposals(self):
        sampling_algo = self.algorithms["sampling"]["algorithm"]
        sampling_algo.settings['nlive'] = self.n_live
        sampling_algo.settings['nproposals'] = self.n_proposals

    def sort_parameters_samples(self, sort_key, ascending):
        if ascending:
            sorted_samples = sorted(self.problem['parameters_samples'],
                                    key=lambda k: k[sort_key], reverse=False)
        else:
            sorted_samples = sorted(self.problem['parameters_samples'],
                                    key=lambda k: k[sort_key], reverse=True)
        self.problem['parameters_samples'] = sorted_samples

    def get_design_vars_bounds(self, which, as_array):
        assert isinstance(which, str), "'which' must be a string."
        permitted_keys = ['lower', 'upper', 'both']
        assert which in permitted_keys, \
            "'which' must be one of the following: " \
            "['lower', 'upper', 'both']. Look for typos or white spaces."
        assert isinstance(as_array, bool), "'as_array' must be boolean."

        design_vars = self.problem["design_variables"]
        if which == "lower":
            lbs = []
            for i, des_var in enumerate(design_vars):
                for k, v in des_var.items():
                    lbs.append(copy.deepcopy(v[0]))
            if as_array:
                lbs = np.asarray(lbs)
            return lbs
        elif which == "upper":
            ubs = []
            for i, des_var in enumerate(design_vars):
                for k, v in des_var.items():
                    ubs.append(copy.deepcopy(v[1]))
            if as_array:
                ubs = np.asarray(ubs)
            return ubs
        elif which == "both":
            lbs, ubs = [], []
            for i, des_var in enumerate(design_vars):
                for k, v in des_var.items():
                    lbs.append(copy.deepcopy(v[0]))
                    ubs.append(copy.deepcopy(v[1]))
            if as_array:
                lbs, ubs = np.asarray(lbs), np.asarray(ubs)
            return lbs, ubs

    # Procedures for handling # of live points and # of proposals
    def obtain_beta_nlive_nproposals(self):
        if self.phase == "INITIAL":
            spec = self.settings["phases_setup"]["initial"]
            self.beta = None
            self.n_live = spec["nlive"]
            self.n_proposals = spec["nproposals"]
        elif self.phase in ["DETERMINISTIC", "NMVP_SEARCH"]:
            pass
        elif self.phase == "TRANSITION":
            pass
        elif self.phase == "PROBABILISTIC":
            spec = \
                self.settings["phases_setup"]["probabilistic"]["nlive_change"]
            mode = spec["mode"]
            if mode == "user_given":
                schedule = spec["schedule"]
                for i, configuration in enumerate(schedule):
                    b, n, r = configuration
                    if i + 1 < len(schedule):
                        a_next, n_next, r_next = schedule[i + 1]
                        if b <= self.worst_phi < a_next:
                            self.beta, self.n_live, self.n_proposals = b, n, r
                            break
                    else:
                        self.beta, self.n_live, self.n_proposals = b, n, r
                        break
            else:
                assert False, "Not implemented yet"

        else:
            assert False, "Unrecognized phase."

    def do_one_sampling_round(self):
        salgo = self.algorithms['sampling']['algorithm']
        sampling_run_status = salgo.run()
        possible_status = ['SUCCESS', 'STOPPED']
        assert sampling_run_status in possible_status, \
            "Got unrecognised status from the sampling algorithm."

        if sampling_run_status == "SUCCESS":
            self.solver_iteration += 1

        elif sampling_run_status == "STOPPED":
            self.solver_iteration += 1
            self.status = "FAILED"

        else:
            assert False, "Unrecognized sampling status."

    # The sorting function for NS algorithm
    def phi(self, d_mat):
        if self.phase == "INITIAL":
            npe = int(len(d_mat))
            nme = npe
            nr = 0
            t0 = time.time()
            fvalues, g_vec_list, self.g_dim = \
                self._score_evaluator.evaluate(d_mat)
            dt_eval = time.time() - t0
            dt_prop = 0.0
            self.update_main_info(npe, nme, nr, dt_prop, dt_eval)

        elif self.phase in ["DETERMINISTIC", "NMVP_SEARCH"]:
            npe = int(len(d_mat))
            nme = npe
            t0 = time.time()
            fvalues, g_vec_list, self.g_dim = \
                self._score_evaluator.evaluate(d_mat)
            dt_eval = time.time() - t0

            salgo = self.algorithms['sampling']['algorithm']
            dpoints = salgo.get_dead_points()
            if dpoints is None:
                nr = 0
            else:
                nr = int(len(dpoints))

            dt_prop = salgo.run_details["cpu_secs_for_proposals"]

            self.update_main_info(npe, nme, nr, dt_prop, dt_eval)

            if self.settings["score_evaluation"]["store_constraints"]:
                p_best = self.problem["parameters_best_estimate"]
                for i, g_vec in enumerate(g_vec_list):
                    item = {'d': d_mat[i, :].tolist(),
                            'p': p_best,
                            'g': g_vec.tolist()}
                    self.g_info.append(item)

        elif self.phase == "TRANSITION":
            npe = int(len(d_mat))
            nr = 0
            t0 = time.time()
            fvalues, nme, g_mat_list = self._efp_evaluator.evaluate(d_mat)
            dt_eval = time.time() - t0
            dt_prop = 0.0
            self.update_main_info(npe, nme, nr, dt_prop, dt_eval)

        elif self.phase == "PROBABILISTIC":
            if self.topup_todo is True:
                npe = len(d_mat)

                salgo = self.algorithms['sampling']['algorithm']
                dt_prop = salgo.run_details["cpu_secs_for_proposals"]

                t0 = time.time()
                fvalues, nme, g_mat_list = self._efp_evaluator.evaluate(d_mat)
                dt_eval = time.time() - t0

                self.update_topup_info(npe, nme, dt_prop, dt_eval)

            else:
                npe = int(len(d_mat))

                t0 = time.time()
                fvalues, nme, g_mat_list = self._efp_evaluator.evaluate(d_mat)
                dt_eval = time.time() - t0

                salgo = self.algorithms['sampling']['algorithm']
                dpoints = salgo.get_dead_points()
                if dpoints is None:
                    nr = 0
                else:
                    nr = int(len(dpoints))

                dt_prop = salgo.run_details["cpu_secs_for_proposals"]

                self.update_main_info(npe, nme, nr, dt_prop, dt_eval)

                if self.settings["efp_evaluation"]["store_constraints"]:
                    p_samples = self.problem["parameters_samples"]
                    for i, g_mat in enumerate(g_mat_list):
                        d_vec = d_mat[i, :].tolist()
                        for j, g_vec in enumerate(g_mat):
                            item = {'d': d_vec,
                                    'p': p_samples[j]['c'],
                                    'g': g_vec.tolist()}
                            self.g_info.append(item)
        else:
            assert False, "The phi function of the phase not found."

        return fvalues

    def obtain_worst_phi(self):
        salgo = self.algorithms['sampling']['algorithm']
        self.worst_phi = salgo.get_live_points()[0].f

    def reset_main_info(self):
        self.main_info = {
            "n_phi_evals": 0,
            "n_model_evals": 0,
            "n_replacements_done": 0,
            "cpu_secs": {
                "proposing": 0.0,
                "evaluating": 0.0
            }
        }

    def update_main_info(self, npe, nme, nr, dt_prop, dt_eval):
        self.main_info = {
            "n_phi_evals": self.main_info["n_phi_evals"] + npe,
            "n_model_evals": self.main_info["n_model_evals"] + nme,
            "n_replacements_done": nr,
            "cpu_secs": {
                "proposing":
                    self.main_info["cpu_secs"]["proposing"] + dt_prop,
                "evaluating":
                    self.main_info["cpu_secs"]["evaluating"] + dt_eval
            }
        }
        return None

    def reset_topup_info(self):
        self.topup_info = {
            "n_phi_evals": 0,
            "n_model_evals": 0,
            "cpu_secs": {
                "proposing": 0.0,
                "phi_eval": 0.0
            }
        }

    def update_topup_info(self, npe, nme, dt_prop, dt_eval):
        self.topup_info = {
            "n_phi_evals": self.topup_info["n_phi_evals"] + npe,
            "n_model_evals": self.topup_info["n_model_evals"] + nme,
            "cpu_secs": {
                "proposing":
                    self.topup_info["cpu_secs"]["proposing"] + dt_prop,
                "phi_eval":
                    self.topup_info["cpu_secs"]["phi_eval"] + dt_eval
            }
        }
        return None

    # Phase changing checks
    def compute_frac_live_inside_nominal_ds(self):
        sampling_algo = self.algorithms["sampling"]["algorithm"]
        lpts = sampling_algo.get_live_points()

        score_type = self.settings["score_evaluation"]["score_type"]
        if score_type == "indicator":
            inside = np.array([point.f >= 1.0 for point in lpts])
        elif score_type == "sigmoid":
            threshold = -self.g_dim * np.log(2)
            inside = np.array([point.f >= threshold for point in lpts])
        else:
            assert False, "unrecognized score type."

        n_inside = np.sum(inside)
        self.frac_live_in_nominal_ds = n_inside / float(self.n_live)

    def compute_frac_live_inside_target_pds(self):
        sampling_algo = self.algorithms["sampling"]["algorithm"]
        lpoints = sampling_algo.get_live_points()

        nlive_inside = 0
        alpha = self.problem['target_reliability']
        for lpoint in lpoints:
            if lpoint.f >= alpha:
                nlive_inside += 1

        spec = self.settings["phases_setup"]["probabilistic"]["nlive_change"]
        mode = spec["mode"]
        if mode == "user_given":
            schedule = spec["schedule"]
            n_list = [item[1] for item in schedule]
            nmax = max(n_list)
            self.frac_live_inside_target_pds = float(nlive_inside) / float(nmax)
        else:
            assert False, "Not implemented yet."

    # Progress printing
    def print_progress_summary(self):
        print("Solver iteration:", self.solver_iteration)
        print("\t *Phase:", self.phase)

        if self.phase == "INITIAL":
            print("\t *lowest F value: %.5f "
                  "| # live points: %d "
                  % (self.worst_phi, self.n_live))
        elif self.phase in ["NMVP_SEARCH", "FINAL"]:
            pass
        elif self.phase == "DETERMINISTIC":
            print("\t *lowest F value: %.5f "
                  "| # live points: %d "
                  "| # proposals: %d"
                  % (self.worst_phi, self.n_live, self.n_proposals))
            print("\t *Fraction of live points in DS~nominal: %.4f"
                  % self.frac_live_in_nominal_ds)
        elif self.phase == "TRANSITION":
            print("\t *lowest F value: %.5f "
                  "| # live points: %d "
                  % (self.worst_phi, self.n_live))
        elif self.phase == "PROBABILISTIC":
            print("\t *lowest F value: %.5f "
                  "| # live points: %d "
                  "| # proposals: %d"
                  % (self.worst_phi, self.n_live, self.n_proposals))
            print("\t *Fraction of live points in DS~%.2f%%: %.4f"
                  % (self.problem['target_reliability']*100,
                     self.frac_live_inside_target_pds))
        else:
            assert False, "Unrecognized phase."

    # Procedures for output handling
    def collect_output(self):
        if self.phase == "INITIAL":
            salgo = self.algorithms['sampling']['algorithm']
            lpoints = salgo.get_live_points()

            the_container = dict()
            the_container.update({
                "phase": self.phase,
                "samples": lpoints,
                "constraints_info": self.g_info,
                "performance": {
                    "n_evals": {
                        "phi": self.main_info["n_phi_evals"],
                        "model": self.main_info["n_model_evals"]
                    },
                    "cpu_time": {
                        "uom": "seconds",
                        "evaluating": self.main_info["cpu_secs"]["evaluating"]
                    }
                }
            })
            self.output_buffer.append(the_container)

        elif self.phase == "DETERMINISTIC":
            salgo = self.algorithms['sampling']['algorithm']
            dpoints = salgo.get_dead_points()

            the_container = dict()
            the_container.update({
                "iteration": self.solver_iteration,
                "phase": self.phase,
                "samples": dpoints,
                "constraints_info": self.g_info,
                "performance": {
                    "n_evals": {
                        "phi": self.main_info["n_phi_evals"],
                        "model": self.main_info["n_model_evals"]
                    },
                    "n_replacements_done": self.main_info["n_replacements_done"],
                    "cpu_time": {
                        "uom": "seconds",
                        "proposing": self.main_info["cpu_secs"]["proposing"],
                        "evaluating": self.main_info["cpu_secs"]["evaluating"],
                        "iteration": self.cpu_secs_for_iteration
                    }
                }
            })

            if self.status in ["FINISHED", "FINISHED_DETERMINISTIC_PHASE"]:
                lpoints = salgo.get_live_points()
                the_container["samples"].extend(lpoints)

            self.output_buffer.append(the_container)

        elif self.phase == "NMVP_SEARCH":
            pass

        elif self.phase == "TRANSITION":
            salgo = self.algorithms['sampling']['algorithm']
            lpoints = salgo.get_live_points()
            the_container = dict()
            the_container.update({
                "phase": self.phase,
                "samples": lpoints,
                "constraints_info": self.g_info,
                "performance": {
                    "n_evals": {
                        "phi": self.main_info["n_phi_evals"],
                        "model": self.main_info["n_model_evals"]
                    },
                    "cpu_time": {
                        "uom": "seconds",
                        "evaluating": self.main_info["cpu_secs"]["evaluating"]
                    }
                }
            })
            self.output_buffer.append(the_container)

        elif self.phase == "PROBABILISTIC":
            salgo = self.algorithms['sampling']['algorithm']
            dpoints = salgo.get_dead_points()

            the_container = dict()
            the_container.update({
                "iteration": self.solver_iteration,
                "phase": self.phase,
                "samples": dpoints,
                "constraints_info": self.g_info,
                "performance": {
                    "n_evals": {
                        "phi": {
                            "main": self.main_info["n_phi_evals"],
                            "topup": self.topup_info["n_phi_evals"]
                        },
                        "model": {
                            "main": self.main_info["n_model_evals"],
                            "topup": self.topup_info["n_model_evals"]
                        },
                    },
                    "n_replacements_done": self.main_info["n_replacements_done"],
                    "cpu_time": {
                        "uom": "seconds",
                        "proposing": {
                            "main": self.main_info["cpu_secs"]["proposing"],
                            "topup": self.topup_info["cpu_secs"]["proposing"],
                        },
                        "evaluating": {
                            "main": self.main_info["cpu_secs"]["evaluating"],
                            "topup": self.topup_info["cpu_secs"]["phi_eval"]
                        },
                        "iteration": self.cpu_secs_for_iteration
                    }
                }
            })

            if self.status == "FINISHED":
                lpoints = salgo.get_live_points()
                the_container["samples"].extend(lpoints)

            self.output_buffer.append(the_container)

        else:
            assert False, "Unrecognized phase."

        return None

    def clear_output_buffer(self):
        self.output_buffer = []

    def attach(self, o):
        self.observers.append(o)

    def detach(self, o):
        self.observers.pop(o)

    def notify_observers(self):
        for o in self.observers:
            o.update()

    # Post-solve steps
    def do_post_solve_steps(self, om):
        print('Solver has no post-solve steps to do.')
