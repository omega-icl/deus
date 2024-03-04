import numpy as np

from deus.activities.solvers.evaluators import \
    ExpensiveFunctionEvaluator
from deus.activities.solvers.evaluators.smescores.data_handler import\
    SMEScoreEvalDataHandler
from deus.activities.solvers.evaluators.smescores.script_handler import\
    SMEScoreEvalScriptHandler


class SMEScoreEvaluator(ExpensiveFunctionEvaluator):
    def __init__(self, info, eval_method, eval_options,
                 e_bounds, spread_to_error_bound):
        super().__init__(info)

        eval_path = self.ufunc_script_path
        self._data_handler = SMEScoreEvalDataHandler(eval_path,
                                                     e_bounds,
                                                     spread_to_error_bound)

        script_handler_info = {
            'eval_path': self.ufunc_script_path,
            'ufunc_script_name': self.ufunc_script_name,
            'ufunc_name': self.ufunc_name
        }
        self._script_handler = \
            SMEScoreEvalScriptHandler(script_handler_info,
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
        assert self._eval_method is not None, \
            "First specify evaluation method"
        assert isinstance(eval_options, dict), \
            "evaluation options must be a dict"

        if self._eval_method == "serial":
            mkeys = ['ufunc_ptr']
            assert all(mkey in eval_options.keys() for mkey in mkeys),\
                "eval options keys must be:\n" \
                "'ufunc_ptr'"

        elif self._eval_method == "mppool":
            mkeys = ['pool_size']
            assert all(mkey in eval_options.keys() for mkey in mkeys)
            self._script_handler.eval_options = eval_options

        elif self._eval_method == "mpi":
            assert False, "Not implemented yet"

        self._eval_options = eval_options
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

        elif self._eval_method == "mpi":
            assert False, "not implemented yet."

        return func_values

    def _expensive_func(self, inputs):
        '''
        :param inputs: array MxD, each row is a point in P-space
        :return:
            * score_values: 1d array of M scalars that are score(s);
        '''
        p_mat = inputs
        e_bounds = np.array(self._data_handler.get_e_bounds())
        e_s2b_ratio = self._data_handler.get_spread_to_error_bound()

        p_shape = np.shape(p_mat)
        if len(p_shape) == 1:
            p_num, p_dim = 1, p_shape
        else:
            p_num, p_dim = p_shape

        normalization_vec = e_s2b_ratio/e_bounds
        e_mat_bounds = np.array([e_bounds]*p_num)

        e_mat = self._eval_options['ufunc_ptr'](p_mat)

        abs_mat = np.abs(e_mat)
        diff_mat = abs_mat - e_mat_bounds
        max_mat = np.maximum(diff_mat, 0.0)
        prod_mat = normalization_vec*max_mat
        final_mat = prod_mat*prod_mat
        score_values = -0.5*np.sum(final_mat, axis=1)

        nans = np.isnan(score_values)
        for i, is_a_nan in enumerate(nans):
            if is_a_nan:
                score_values[i] = -np.inf
        return score_values

    def _evaluate_using_script(self, inputs):
        self._data_handler.set_inputs(inputs)
        self._data_handler.write_eval_data()

        if self.must_update_eval_script:
            self._script_handler.write_eval_script()
            self.must_update_eval_script = False

        the_script_fpne = self._script_handler.get_script_fpne()
        self._execute_script(the_script_fpne)
        self._data_handler.read_eval_data()
