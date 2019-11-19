import copy
import numpy as np
import time

from deus import utils
from deus.activities.interfaces import Subject, Observer
from deus.activities.solvers.algorithms.points import BayesPoint
from deus.activities.solvers.algorithms.composite.factory import \
    CompositeAlgorithmFactory
from deus.activities.solvers.algorithms.composite import \
    NestedSamplingWithGlobalSearch
from deus.activities.solvers.evaluators.dsscore_evaluator import \
    DSScoreEvaluator
from deus.activities.solvers.evaluators.efp_evaluator import \
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

        self.observers = []
        self.solver_iteration = 0
        self.n_proposals = 0
        self.n_model_evals = 0
        self.cpu_secs_for_iteration = 0.0
        self.cpu_secs_for_evaluation = 0.0
        self.g_info = []

        self.output_buffer = []

        # state
        self.status = "READY"
        self.phase = "INITIAL"
        self.worst_alpha = 0.0
        self.a, self.n, self.r = None, None, None

    def set_problem(self, problem):
        # Check is already done by the activity manager.
        self.problem = problem

    def set_settings(self, settings):
        assert isinstance(settings, dict), \
            "The settings must be a dictionary."

        mkeys = ['score_evaluation', 'efp_evaluation', 'points_schedule',
                 'stop_criteria']
        assert all(mkey in settings.keys() for mkey in mkeys), \
            "The 'settings' keys must be the following:\n" \
            "'score_evaluation', 'efp_evaluation', 'points_schedule'," \
            "'stop_criteria'. Look for typos, white spaces or missing keys."


        score_eval = settings['score_evaluation']
        assert isinstance(score_eval, dict), \
            "'score_evaluation' must be a dictionary."

        if score_eval["method"] == "serial":
            mkeys = ['method', 'constraints_func_ptr', 'store_constraints']
            assert all(mkey in score_eval.keys() for mkey in mkeys), \
                "The 'score_evaluation' keys must be the following:\n" \
                "'method', 'constraints_func_ptr', 'store_constraints'." \
                "Look for typos, white spaces or missing keys."

        elif score_eval["method"] == "mppool":
            mkeys = ['method', 'pool_size', 'store_constraints']
            assert all(mkey in score_eval.keys() for mkey in mkeys), \
                "The 'score_evaluation' keys must be the following:\n" \
                "'method', 'pool_size', 'store_constraints'." \
                "Look for typos, white spaces or missing keys."
            assert isinstance(score_eval['pool_size'], int), \
                "'n_processes' must be an integer."
            n_procs = score_eval['pool_size']
            assert (n_procs == -1 or n_procs >= 2), \
                "'pool_size' must be >=2 or -1.\n"\
                "-1: # processes = # logical cores."

        elif score_eval["method"] == "mpi":
            assert False, "Not implemented yet."

        else:
            assert False, "'score_evaluation' method not recognized."


        efp_eval = settings['efp_evaluation']
        assert isinstance(efp_eval, dict), \
            "'efp_evaluation' must be a dictionary."

        if efp_eval["method"] == "serial":
            mkeys = ['method', 'constraints_func_ptr', 'store_constraints',
                     'acceleration']
            assert all(mkey in efp_eval.keys() for mkey in mkeys), \
                "The 'efp_evaluation' keys must be the following:\n" \
                "'method', 'constraints_func_ptr', 'store_constraints', " \
                "'acceleration'. Look for typos, white spaces or missing keys."

        elif efp_eval["method"] == "mppool":
            # assert False, "Not implemented yet."
            mkeys = ['method', 'pool_size', 'store_constraints', 'acceleration']
            assert all(mkey in efp_eval.keys() for mkey in mkeys), \
                "The 'efp_evaluation' keys must be the following:\n" \
                "'method', 'pool_size', 'store_constraints', 'acceleration'."\
                "Look for typos, white spaces or missing keys."
            assert isinstance(efp_eval['pool_size'], int), \
                "'n_processes' must be an integer."
            n_procs = efp_eval['pool_size']
            assert (n_procs == -1 or n_procs >= 2), \
                "'pool_size' must be >=2 or -1.\n"\
                "-1: # processes = # logical cores."

        elif efp_eval["method"] == "mpi":
            assert False, "Not implemented yet."

        else:
            assert False, "'efp_evaluation' method not recognized."


        pts_schedule = settings['points_schedule']
        assert isinstance(pts_schedule, list), \
            "'points_schedule' must be a list."
        for i, item in enumerate(pts_schedule):
            assert isinstance(item, tuple), \
                "'points_schedule' must contain (a, n, r) tuples, where:" \
                "a - reliability level in [0, 1]; " \
                "n - number of live points; " \
                "r - number of replacements attempts per iteration."
            if len(pts_schedule) > 1 and i < len(pts_schedule) - 1:
                n1, n2 = pts_schedule[i][1], pts_schedule[i+1][1]
                assert (n1 < n2), \
                    "The number of live points must always increase."

        self.settings = settings

    def set_algorithms(self, algos):
        assert self.problem is not None, \
            "Define 'problem' before the algorithms of 'ds_solver'."
        assert self.settings is not None, \
            "Define 'settings' before the algorithms of 'ds_solver'."
        assert isinstance(algos, dict), \
            "algorithms must be a dictionary."

        mkeys = ['sampling']
        assert all(mkey in algos.keys() for mkey in mkeys), \
            "The 'algorithms' of 'pe_solver' should be for the steps:\n" \
            "['sampling']. " \
            "Look for typos, white spaces or missing keys."

        algos_permitted = [NestedSamplingWithGlobalSearch.get_type()
                           + "-" +
                           NestedSamplingWithGlobalSearch.get_ui_name()]

        sampling_algo_name = algos["sampling"]["algorithm"]
        assert sampling_algo_name in algos_permitted, \
            "The algorithm specified for 'sampling' is not permitted."

        permitted_stop_criteria = ['inside_fraction']

        specified_stop_criteria = utils.keys_in(
            self.settings["stop_criteria"])
        for ssc in specified_stop_criteria:
            assert ssc in permitted_stop_criteria,\
                "Stop_criteria '" \
                + ssc + "' is not permitted in this context."

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
        }
        eval_method = score_eval["method"]
        if eval_method == "serial":
            eval_options = {
                'ufunc_ptr': score_eval["constraints_func_ptr"],
                'store_constraints': score_eval["store_constraints"]
            }

        elif eval_method == "mppool":
            # assert False, "Not implemented yet."
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
                'store_constraints': efp_eval["store_constraints"],
                'acceleration': efp_eval["acceleration"]
            }

        elif eval_method == "mppool":
            # assert False, "Not implemented yet."
            eval_options = {
                'pool_size': efp_eval["pool_size"],
                'store_constraints': efp_eval["store_constraints"],
                'acceleration': efp_eval["acceleration"]
            }

        elif eval_method == "mpi":
            assert False, "Not implemented yet."

        p_samples = self.problem["parameters_samples"]
        self._efp_evaluator = EFPEvaluator(evaluator_info,
                                           eval_method,
                                           eval_options,
                                           p_samples)

    def solve(self):
        if self.phase == "INITIAL":
            self.sort_parameters_samples(sort_key='w', ascending=False)
            self.p_coords, self.p_weights = \
                self.get_parameters_samples_coords_and_weights()
            self._p_num, self._p_dims = np.shape(self.p_coords)
            self._inside_frac = None

            sampling_algo = self.algorithms["sampling"]["algorithm"]
            lbs, ubs = self.get_design_vars_bounds(which="both", as_array=True)
            sampling_algo.set_bounds(lbs, ubs)
            sampling_algo.set_sorting_func(self.phi)

            self.a, self.n, self.r = self.settings['points_schedule'][0]
            sampling_algo.settings['nlive'] = self.n
            sampling_algo.settings['nreplacements'] = self.r
            self.phase = "DETERMINISTIC"
            sampling_algo.set_sorting_func(self.phi)
            sampling_algo.initialize_live_points()
            print("Phase INITIAL is over.")

        while (self.phase == "DETERMINISTIC"):
            self.do_one_sampling_round()
            if self.is_deterministic_phase_over():
                print("Phase DETERMINISTIC is over.")
                self.phase = "TRANSITION"
                break

        if self.phase == "TRANSITION":
            self.solver_iter_phase1_ended = self.solver_iteration
            self.phase = "PROBABILISTIC"
            self.worst_alpha = 0.0
            self._efp_evaluator.set_worst_efp(self.worst_alpha)
            sampling_algo.evaluate_live_points_fvalue()
            sampling_algo.live_points.sort()
            self.worst_alpha = sampling_algo.get_live_points()[0].f
            self._efp_evaluator.set_worst_efp(self.worst_alpha)
            print("Phase TRANSITION is over.")

        while(self.phase == "PROBABILISTIC"):
            if self.status == "SAMPLING_SUCCEDED":
                worst = sampling_algo.get_live_points()[0].f
                self.worst_alpha = copy.deepcopy(worst)
                self._efp_evaluator.set_worst_efp(self.worst_alpha)
                self.set_live_points_according_schedule()
                self.do_one_sampling_round()
            elif self.status == "SAMPLING_FAILED":
                print("Sampling algorithm failed.")
                break
            elif self.status == "SUBALGORITHM_STOPPED":
                print("A subalgorithm finished.")
                break
            elif self.status == "FINISHED":
                print("Phase PROBABILISTIC is over.")
                print("Solver finished solving the problem.")
                break
            else:
                assert False, "Unrecognizable solver status."

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

    def sort_parameters_samples(self, sort_key, ascending):
        if ascending:
            sorted_samples = sorted(self.problem['parameters_samples'],
                                    key=lambda k: k[sort_key], reverse=False)
        else:
            sorted_samples = sorted(self.problem['parameters_samples'],
                                    key=lambda k: k[sort_key], reverse=True)
        self.problem['parameters_samples'] = sorted_samples

    def get_parameters_samples_coords_and_weights(self):
        p_samples = self.problem['parameters_samples']
        p_num = len(p_samples)
        p_dims = len(p_samples[0]['c'])

        p_coords = np.ndarray((p_num, p_dims))
        p_weights = np.ndarray(p_num)
        for i, sample in enumerate(p_samples):
            p_coords[i, :] = sample['c']
            p_weights[i] = sample['w']
        return p_coords, p_weights

    def set_live_points_according_schedule(self):
        pts_schedule = self.settings["points_schedule"]
        sampling_algo = self.algorithms['sampling']['algorithm']
        for i, configuration in enumerate(pts_schedule):
            a, n, r = configuration
            if i+1 < len(pts_schedule):
                a_next, n_next, r_next = pts_schedule[i + 1]
                if a <= self.worst_alpha < a_next:
                    self.a, self.n, self.r = a, n, r
                    break
            else:
                self.a, self.n, self.r = a, n, r
                break
        sampling_algo.top_up_to(self.n)
        sampling_algo.settings["nreplacements"] = self.r

    def do_one_sampling_round(self):
        sampling_algo = self.algorithms['sampling']['algorithm']
        t0 = time.time()
        sampling_run_status = sampling_algo.run()
        self.cpu_secs_for_iteration = time.time() - t0
        possible_status = ['SUCCESS', 'STOPPED']
        assert sampling_run_status in possible_status, \
            "Got unrecognised status from the sampling algorithm."

        if sampling_run_status == "SUCCESS":
            self.solver_iteration += 1
            the_container = dict()
            self.collect_iteration_output(the_container)
            self.g_info = []  # Always clear this after output collection
            finished_solve, c = self.is_any_stop_criterion_met()
            if finished_solve:
                self.status = "FINISHED"
                print(self.settings["stop_criteria"][c], "is fulfilled.")
                # now we must add the live points to samples
                lpoints = sampling_algo.get_live_points()
                the_container["samples"].extend(lpoints)
            else:
                self.status = "SAMPLING_SUCCEDED"

            self.output_buffer.append(the_container)
            self.print_progress()
            self.notify_observers()

        elif sampling_run_status == "STOPPED":
            self.solver_iteration += 1
            the_container = dict()
            self.collect_iteration_output(the_container)
            self.g_info = []  # Always clear this after output collection
            self.output_buffer.append(the_container)

            self.status = "SUBALGORITHM_STOPPED"
            self.notify_observers()

        else:
            self.status = "SAMPLING_FAILED"
            assert False, "Not implemented yet."

    # The sorting function for NS algorithm
    def phi(self, d_mat):
        if self.phase == "DETERMINISTIC":
            t0 = time.time()
            fvalues, g_vec_list = self._score_evaluator.evaluate(d_mat)
            self.cpu_secs_for_evaluation = time.time() - t0
            self.n_proposals = len(d_mat)
            self.n_model_evals = len(d_mat)

            # Do something with g_vec_list TODO
            if self.settings["score_evaluation"]["store_constraints"]:
                p_best = self.problem["parameters_best_estimate"]
                for i, g_vec in enumerate(g_vec_list):
                    item = {'d': d_mat[i, :].tolist(),
                            'p': p_best,
                            'g': g_vec.tolist()}
                    self.g_info.append(item)

        elif self.phase == "PROBABILISTIC":  # WIP: EFP Evaluator
            t0 = time.time()
            fvalues, nme, g_mat_list = self._efp_evaluator.evaluate(d_mat)
            self.cpu_secs_for_evaluation = time.time() - t0
            self.n_proposals = len(d_mat)
            self.n_model_evals = nme

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

    # Phase changing checks
    def is_deterministic_phase_over(self):
        sampling_algo = self.algorithms["sampling"]["algorithm"]
        lpts = sampling_algo.get_live_points()
        inside = np.array([point.f >= 1.0 for point in lpts])
        return np.all(inside)

    # Output Handling
    def print_progress(self):
        print("Solver iteration:", self.solver_iteration)

        print("\t *Phase:", self.phase)

        print("\t *lowest F value: %.5f "
              "| live points: %d "
              "| replacement attempts: %d"
              %(self.worst_alpha, self.n, self.r))

        print("\t *Fraction of live points in DS~%.2f%%: %.4f"
              % (self.problem['target_reliability']*100, self._inside_frac))

    def attach(self, o):
        self.observers.append(o)

    def detach(self, o):
        self.observers.pop(o)

    def notify_observers(self):
        for o in self.observers:
            o.update()

    def clear_output_buffer(self):
        self.output_buffer = []

    def collect_iteration_output(self, container):
        sampling_algo = self.algorithms['sampling']['algorithm']
        dpoints = sampling_algo.get_dead_points()
        run_details = sampling_algo.run_details

        container.update({
            "iteration": self.solver_iteration,
            "phase": self.phase,
            "samples": dpoints,
            "constraints_info": self.g_info,
            "performance": {
                "n_phi_evaluations": self.n_proposals,
                "n_model_evaluations": self.n_model_evals,
                "n_replacements": len(dpoints),
                "cpu_secs": {
                    "proposals_generation":
                        run_details["cpu_secs_for_proposals"],
                    "phi_evaluation": self.cpu_secs_for_evaluation,
                    "total": self.cpu_secs_for_iteration
                }
            }
        })

    def is_any_stop_criterion_met(self):
        for c, criterion in enumerate(self.settings["stop_criteria"]):
            for k, v in criterion.items():
                if k == "inside_fraction":
                    fraction_wanted = v

                    sampling_algo = self.algorithms["sampling"]["algorithm"]
                    lpoints = sampling_algo.get_live_points()

                    nlive = len(lpoints)
                    num_of_lpoints_inside = 0
                    alpha = self.problem['target_reliability']
                    for lpoint in lpoints:
                        if lpoint.f >= alpha:
                            num_of_lpoints_inside += 1

                    pts_schedule = self.settings['points_schedule']
                    n_list = [item[1] for item in pts_schedule]
                    nlive_max = max(n_list)
                    self._inside_frac = num_of_lpoints_inside / float(nlive_max)

                    if self._inside_frac >= fraction_wanted:
                        return True, c
        return False, 0

    # Post-solve steps
    def do_post_solve_steps(self, om):
        print('Solver has no post-solve steps to do.')
