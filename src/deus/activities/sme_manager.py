import pickle

from deus.utils.assertions import DEUS_ASSERT

from deus.activities import ActivityManager
from deus.activities.output import SetMembershipEstimationOutputManager
from deus.activities.solvers.factory import SolverFactory


class SetMembershipEstimationManager(ActivityManager):
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
        self.output_manager = \
            SetMembershipEstimationOutputManager(self.cs_folder)

    def check_activity_type(self, a_type):
        assert a_type == "sme", \
            "The activity type must be \"sme\". Recheck the activity form."

    def check_problem(self, problem):
        assert isinstance(problem, dict), \
            "'problem' must be a dictionary."

        mkeys = ['user_script_filename', 'errors_func_name', 'errors_bound',
                 'parameters']
        DEUS_ASSERT.has(mkeys, problem, "problem")

        errors_bound = problem["errors_bound"]
        assert isinstance(errors_bound, list), \
            "'errors_bound', must be a list of positive real numbers."

        parameters = problem["parameters"]
        assert isinstance(parameters, list), \
            "'parameters' must be a list of dictionaries."
        for i, item in enumerate(parameters):
            assert isinstance(item, dict), \
                "All items of 'parameters' are dictionaries."
            assert len(item.keys()) == 1, \
                "'Items in 'parameters' must be a single key-value dictionary."
            for k, v in item.items():
                assert isinstance(v, list), \
                    "The value of any 'item' in 'parameters' must be as " \
                    "[<lower_bound>, <upper_bound>]."
                assert len(v) == 2, \
                    "A parameter must have specified exactly one lbound and " \
                    "one ubound."
                assert v[0] < v[1], \
                    "Bad input: The lbound > ubound for parameter '" \
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
                self.output_manager.write_performance_summary()

    def is_time_to_save(self):
        if self.solver.status in ["FINISHED", "SUBALGORITHM_STOPPED"]:
            return True

        save_period = self.settings["save_period"]
        solver_iter = self.solver.solver_iteration
        return solver_iter % save_period == 0

    def save_solution_state(self):
        with open(self.cs_folder + 'solution_state.pkl', 'wb') as file:
            pickle.dump(self.__dict__, file)
