import pickle

from deus.activities import ActivityManager
from deus.activities.output import DesignSpaceOutputManager
from deus.activities.solvers.factory import SolverFactory


class DesignSpaceManager(ActivityManager):
    def __init__(self, activity_form):
        super().__init__()
        self.check_activity_type(activity_form["activity_type"])

        self.settings = None
        settings = activity_form["activity_settings"]
        self.check_settings(settings)
        self.settings = settings

        problem = activity_form["problem"]
        self.check_problem(problem)
        self.problem = problem

        cs_path = self.settings["case_path"]
        solver_name = activity_form["solver"]["name"]
        solver_settings = activity_form["solver"]["settings"]
        solver_algorithms = activity_form["solver"]["algorithms"]
        self.solver = SolverFactory.create(solver_name,
                                           cs_path,
                                           self.problem,
                                           solver_settings,
                                           solver_algorithms)

        self.solver.attach(self)

        cs_name = self.settings["case_name"]
        self.cs_folder = cs_path + "/" + cs_name + "/"
        self.output_manager = DesignSpaceOutputManager(self.cs_folder)

    def check_activity_type(self, a_type):
        assert a_type == "ds", \
            "The activity type must be \"ds\". Recheck the activity form."

    def check_problem(self, problem):
        assert isinstance(problem, dict), \
            "'problem' must be a dictionary."

        mkeys = ['user_script_filename', 'constraints_func_name',
                 'parameters_best_estimate', 'parameters_samples',
                 'target_reliability', 'design_variables']
        assert all(mkey in problem.keys() for mkey in mkeys), \
            "The 'problem' keys must be the following:\n" \
            "'user_script_filename', 'constraints_func_name'," \
            "'parameters_best_estimate', 'parameters_samples'," \
            "'target_reliability', 'design_variables'." \
            "Look for typos, white spaces or missing keys."

        assert(0. <= problem['target_reliability'] <= 1.), \
            "'target_reliability' must be in [0, 1]."

        p_best_estimate = problem['parameters_best_estimate']
        assert isinstance(p_best_estimate, list),\
            "'parameters_best_estimate' must be a list."
        assert (len(p_best_estimate) ==
                len(problem['parameters_samples'][0]['c'])), \
            "'parameters_best_estimate' and 'parameters_samples' must have" \
            " the same dimensionality."

        p_samples = problem['parameters_samples']
        assert isinstance(p_samples, list), \
            "'parameters_samples' must be a list of dictionaries."
        for i, p_sample in enumerate(p_samples):
            assert isinstance(p_sample, dict), \
                "the items in 'parameters_samples' must be a dictionary."
            mkeys = ['c', 'w']
            assert all(mkey in p_sample.keys() for mkey in mkeys),\
                "The 'parameters_samples' item keys must be the following:\n"\
                "'c' - a list of coordinates, "\
                "'w' - a scalar reperesenting a weight."\
                "Look for typos, white spaces or missing keys."

        design_variables = problem["design_variables"]
        assert isinstance(design_variables, list), \
            "'design_variables' must be a list of dictionaries."
        for i, item in enumerate(design_variables):
            assert isinstance(item, dict), \
                "All items of 'design_variables' are dictionaries."
            assert len(item.keys()) == 1,\
                "'Items in 'design_variables' must be a single key-value dictionary."
            for k, v in item.items():
                assert isinstance(v, list), \
                    "The value of any item in 'design_variables' must be as "\
                    "[<lower_bound>, <upper_bound>]."
                assert len(v) == 2, \
                    "A design variable must have specified exactly one lbound "\
                    "and one ubound."
                assert v[0] < v[1], \
                    "Bad input: The lbound > ubound for design variable '" \
                    + str(k) + "'."

    def solve_problem(self):
        self.solver.solve()

    def update(self):
        if self.is_time_to_save():
            self.output_manager.add(self.solver.output_buffer)
            self.output_manager.write_to_disk()
            self.solver.clear_output_buffer()
            self.save_solution_state()

            solver_status = self.solver.status
            if solver_status in ["FINISHED", "SUBALGORITHM_STOPPED"]:
                self.solver.do_post_solve_steps(self.output_manager)
                self.output_manager.write_to_disk()

    def is_time_to_save(self):
        if self.solver.status in ["FINISHED", "SUBALGORITHM_STOPPED"]:
            return True

        save_period = self.settings["save_period"]
        solver_iter = self.solver.solver_iteration
        return solver_iter % save_period == 0

    def save_solution_state(self):
        with open(self.cs_folder + 'solution_state.pkl', 'wb') as file:
            pickle.dump(self.__dict__, file)
