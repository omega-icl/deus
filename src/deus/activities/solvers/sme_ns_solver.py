import copy
import numpy as np
import time

from deus.activities.solvers.validators import ValidatorForSMESolver
from deus.activities.interfaces import Subject, Observer
from deus.activities.solvers.algorithms.composite.factory import \
    CompositeAlgorithmFactory
from deus.activities.solvers.algorithms.composite import \
    NestedSamplingWithGlobalSearch
from deus.activities.solvers.evaluators.smescores.evaluator import \
    SMEScoreEvaluator


class SetMembershipEstimationSolverUsingNS(Subject):
    def __init__(self, cs_path, problem=None, settings=None, algorithms=None):
        self.cs_path = cs_path

        self.problem = None
        if problem is not None:
            self.set_problem(problem)

        self.settings = None
        if settings is not None:
            self.set_settings(settings)

        self.algorithms = None
        self.n_live, self.n_proposals = None, None
        if algorithms is not None:
            self.set_algorithms(algorithms)

        self._score_evaluator = None
        self.create_score_evaluator()

        self.observers = []

        self.solver_iteration = 0
        # performance data
        self.main_info = None
        self.cpu_secs_for_iteration = 0.0

        self.output_buffer = []

        # state
        self.status = "NOT_FINISHED"
        self.phase = "INITIAL"

        self.worst_phi = None

        self.frac_live_in_target_set = None

    # Construction
    def set_problem(self, problem):
        # Check is already done by the activity manager.
        self.problem = problem

    def set_settings(self, settings):
        ValidatorForSMESolver.check_settings(settings)
        self.settings = settings

    def set_algorithms(self, algos):
        assert self.problem is not None, \
            "Define 'problem' before the algorithms of 'ds_solver'."
        assert self.settings is not None, \
            "Define 'settings' before the algorithms of 'ds_solver'."
        ValidatorForSMESolver.check_algorithms(algos)
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

        self.n_live = algos["sampling"]["settings"]["nlive"]
        self.n_proposals = algos["sampling"]["settings"]["nproposals"]

    def create_score_evaluator(self):
        assert self.problem is not None, \
            "Problem must be defined before creating the score evaluator."
        assert self.settings is not None, \
            "Settings must be defined before creating the score evaluator."

        score_eval = self.settings["errors_evaluation"]
        evaluator_info = {
            'ufunc_script_path': self.cs_path,
            'ufunc_script_name': self.problem["user_script_filename"],
            'ufunc_name': self.problem["errors_func_name"]
        }
        eval_method = score_eval["method"]
        if eval_method == "serial":
            eval_options = {
                'ufunc_ptr': score_eval["errors_func_ptr"]
            }

        elif eval_method == "mppool":
            eval_options = {
                'pool_size': score_eval["pool_size"]
            }

        elif eval_method == "mpi":
            assert False, "Not implemented yet."

        err_bounds = self.problem["errors_bound"]
        s2eb_ratio = self.settings["spread_to_error_bound"]
        self._score_evaluator = SMEScoreEvaluator(evaluator_info,
                                                  eval_method,
                                                  eval_options,
                                                  err_bounds, s2eb_ratio)

    # The most important procedure :)
    def solve(self):
        if self.phase == "INITIAL":
            self.tell_sampling_algo_the_bounds_and_sorting_function()

            salgo = self.algorithms['sampling']['algorithm']
            self.reset_main_info()
            salgo.initialize_live_points()

            self.obtain_worst_phi()
            self.print_progress_summary()

            self.collect_output()
            self.notify_observers()

            print("Phase INITIAL is over.")
            self.phase = "SEARCH"

        while self.phase == "SEARCH":
            # We assume that live points are sorted ascending by phi!
            t0 = time.time()
            self.reset_main_info()
            self.do_one_sampling_round()
            self.cpu_secs_for_iteration = time.time() - t0

            self.compute_frac_live_inside_target_set()
            self.obtain_worst_phi()
            self.print_progress_summary()
            if self.frac_live_in_target_set == 1.0:
                print("Phase SEARCH is over.")
                self.status = "FINISHED"
                self.collect_output()
                self.notify_observers()
                self.phase = "FINAL"
            elif self.status == "FAILED":
                self.collect_output()
                self.notify_observers()
                print("Failure.")
                self.phase = "FINAL"
            else:
                self.status = "NOT_FINISHED"
                self.collect_output()
                self.notify_observers()
                self.phase = "SEARCH"

        if self.phase == "FINAL":
            print("Set Membership Estimation is done.")

    # Procedures used only during INITIAL phase
    def tell_sampling_algo_the_bounds_and_sorting_function(self):
        lbs, ubs = self.get_parameters_bounds(which="both", as_array=True)
        salgo = self.algorithms["sampling"]["algorithm"]
        salgo.set_bounds(lbs, ubs)
        salgo.set_sorting_func(self.phi)
        return None

    def get_parameters_bounds(self, which, as_array):
        assert isinstance(which, str), "'which' must be a string."
        permitted_keys = ['lower', 'upper', 'both']
        assert which in permitted_keys, \
            "'which' must be one of the following: " \
            "['lower', 'upper', 'both']. Look for typos or white spaces."
        assert isinstance(as_array, bool), "'as_array' must be boolean."

        parameters = self.problem["parameters"]
        if which == "lower":
            lbs = []
            for i, des_var in enumerate(parameters):
                for k, v in des_var.items():
                    lbs.append(copy.deepcopy(v[0]))
            if as_array:
                lbs = np.asarray(lbs)
            return lbs
        elif which == "upper":
            ubs = []
            for i, des_var in enumerate(parameters):
                for k, v in des_var.items():
                    ubs.append(copy.deepcopy(v[1]))
            if as_array:
                ubs = np.asarray(ubs)
            return ubs
        elif which == "both":
            lbs, ubs = [], []
            for i, des_var in enumerate(parameters):
                for k, v in des_var.items():
                    lbs.append(copy.deepcopy(v[0]))
                    ubs.append(copy.deepcopy(v[1]))
            if as_array:
                lbs, ubs = np.asarray(lbs), np.asarray(ubs)
            return lbs, ubs

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
    def phi(self, p_mat):
        if self.phase == "INITIAL":
            npe = int(len(p_mat))
            nme = npe
            nr = 0
            t0 = time.time()
            fvalues = self._score_evaluator.evaluate(p_mat)
            dt_eval = time.time() - t0
            dt_prop = 0.0
            self.update_main_info(npe, nme, nr, dt_prop, dt_eval)

        elif self.phase == "SEARCH":
            npe = int(len(p_mat))
            nme = npe
            t0 = time.time()
            fvalues = self._score_evaluator.evaluate(p_mat)
            dt_eval = time.time() - t0

            salgo = self.algorithms['sampling']['algorithm']
            dpoints = salgo.get_dead_points()
            if dpoints is None:
                nr = 0
            else:
                nr = int(len(dpoints))

            dt_prop = salgo.run_details["cpu_secs_for_proposals"]

            self.update_main_info(npe, nme, nr, dt_prop, dt_eval)

        else:
            assert False, "The phi function of the phase not found."

        return fvalues

    def obtain_worst_phi(self):
        salgo = self.algorithms['sampling']['algorithm']
        self.worst_phi = salgo.get_live_points()[0].f
        return None

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

    # Phase changing checks
    def compute_frac_live_inside_target_set(self):
        sampling_algo = self.algorithms["sampling"]["algorithm"]
        lpts = sampling_algo.get_live_points()

        threshold = 0.0
        inside = np.array([point.f >= threshold for point in lpts])

        n_inside = np.sum(inside)
        self.frac_live_in_target_set = n_inside / float(self.n_live)

    # Progress printing
    def print_progress_summary(self):
        print("Solver iteration:", self.solver_iteration)
        print("\t *Phase:", self.phase)

        if self.phase == "INITIAL":
            print("\t *lowest F value: %.5f "
                  "| # live points: %d "
                  % (self.worst_phi, self.n_live))
        elif self.phase == "SEARCH":
            print("\t *lowest F value: %.5f "
                  "| # live points: %d "
                  "| # proposals: %d"
                  % (self.worst_phi, self.n_live, self.n_proposals))
            print("\t *Fraction of live points in target set: %.4f"
                  % self.frac_live_in_target_set)
        elif self.phase == "FINAL":
            pass
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

        elif self.phase == "SEARCH":
            salgo = self.algorithms['sampling']['algorithm']
            dpoints = salgo.get_dead_points()

            the_container = dict()
            the_container.update({
                "iteration": self.solver_iteration,
                "phase": self.phase,
                "samples": dpoints,
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
