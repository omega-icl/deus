import numpy as np

from deus.activities.solvers.evaluators import ExpensiveFunctionEvaluator
from deus.activities.solvers.evaluators.logl_data_handler import\
    LogLkhdEvalDataHandler
from deus.activities.solvers.evaluators.logl_script_handler import\
    LogLkhdEvalScriptHandler


class LogLkhdEvaluator(ExpensiveFunctionEvaluator):
    def __init__(self, info, eval_method, eval_options):
        super().__init__(info)

        eval_path = self.ufunc_script_path
        self._data_handler = LogLkhdEvalDataHandler(eval_path)

        script_handler_info = {
            'eval_path': self.ufunc_script_path,
            'ufunc_script_name': self.ufunc_script_name,
            'ufunc_name': self.ufunc_name,
        }
        self._script_handler = \
            LogLkhdEvalScriptHandler(script_handler_info,
                                     eval_method,
                                     eval_options,
                                     self._data_handler)

        self._eval_method = None
        self.set_eval_method(eval_method)
        self._eval_options = None
        self.set_eval_options(eval_options)

    def set_eval_method(self, eval_method):
        assert eval_method is not None, "Unspecified evaluation method"
        assert eval_method in ['serial', 'mppool', 'mpi'], \
            "eval_method unrecognised. It must be one of the next" \
            "['serial', 'mppool', 'mpi']"
        self._eval_method = eval_method
        self._script_handler.eval_method = eval_method
        self.must_update_eval_script = True

    def set_eval_options(self, eval_options):
        assert self._eval_method is not None, "First specify evaluation method"
        if self._eval_method == "serial":
            self._eval_options = eval_options

        elif self._eval_method == "mppool":
            self._eval_options = eval_options
            self._script_handler.eval_options = eval_options

        elif self._eval_method == "mpi":
            assert False, "Not implemented yet"

        self.must_update_eval_script = True

    def evaluate(self, inputs):
        if self._eval_method == "serial":
            if self._eval_options['ufunc_ptr'] is None:
                self._evaluate_using_script(inputs)
                the_data = self._data_handler.get_data()
                func_values = the_data['out']
            else:  # Don't use evaluation script
                func_values = self._expensive_func(inputs)

        elif self._eval_method == "mppool":
            self._evaluate_using_script(inputs)
            the_data = self._data_handler.get_data()
            func_values = the_data['out']

        else:
            assert False, "not implemented yet."

        return func_values

    def _expensive_func(self, inputs):
        '''
        :param inputs: array MxP, each row is a point in P-space
        :return:
            * logl_values: 1d array of M scalars that are log-likelihood(s);
        '''

        assert len(np.shape(inputs)) == 2, "'inputs' must be a 2d array."
        logl_values = self._eval_options['ufunc_ptr'](inputs)
        return logl_values

    def _evaluate_using_script(self, inputs):
        self._data_handler.set_inputs(inputs)
        self._data_handler.write_eval_data()

        if self.must_update_eval_script:
            self._script_handler.write_eval_script()
            self.must_update_eval_script = False

        the_script_fpne = self._script_handler.get_script_fpne()
        self._execute_script(the_script_fpne)
        self._data_handler.read_eval_data()
