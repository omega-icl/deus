import numpy as np

from deus.activities.solvers.evaluators import ExpensiveFunctionEvaluator
from deus.activities.solvers.evaluators.dsefp.data_handler import\
    EFPEvalDataHandler
from deus.activities.solvers.evaluators.dsefp.script_handler import\
    EFPEvalScriptHandler


class EFPEvaluator(ExpensiveFunctionEvaluator):
    def __init__(self, info, eval_method, eval_options, p_samples):
        super().__init__(info)

        eval_path = self.ufunc_script_path
        self._data_handler = EFPEvalDataHandler(eval_path, p_samples)

        script_handler_info = {
            'eval_path': self.ufunc_script_path,
            'ufunc_script_name': self.ufunc_script_name,
            'ufunc_name': self.ufunc_name,
        }
        self._script_handler =\
            EFPEvalScriptHandler(script_handler_info,
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
            mkeys = ['ufunc_ptr', 'store_constraints']
            assert all(mkey in eval_options.keys() for mkey in mkeys),\
                "eval options keys must be:\n" \
                "'ufunc_ptr', 'store_constraints'"

        elif self._eval_method == "mppool":
            # assert False, "Not implemented yet"
            mkeys = ['pool_size', 'store_constraints']
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
                n_model_evals = the_data['n_model_evals']
                if self._eval_options['store_constraints']:
                    g_list = the_data['g_list']
                else:
                    g_list = []

            else:  # Don't use evaluation script
                func_values, n_model_evals, g_list = \
                    self._expensive_func(inputs)

        elif self._eval_method == "mppool":
            # assert False, "not implemented yet."
            self._evaluate_using_script(inputs)
            the_data = self._data_handler.get_data()
            func_values = the_data['out']
            n_model_evals = sum(the_data['n_model_evals'])
            g_list = the_data['g_list']

        elif self._eval_method == "mpi":
            assert False, "not implemented yet."

        return func_values, n_model_evals, g_list

    def _expensive_func(self, inputs):
        '''
        :param inputs: array MxD, each row is a point in D-space
        :return:
            * efp_values: list of M scalars that are estimated feasibility
            probabilities;
        '''
        d_mat = inputs
        p_samples = self._data_handler.get_p_samples()
        # WARNING: We assume here that p_weights are normalized already!!!
        must_store_g = self._eval_options['store_constraints']

        d_shape = np.shape(d_mat)
        if len(d_shape) == 1:
            d_num, d_dim = 1, d_shape
        else:
            d_num, d_dim = d_shape

        p_num = len(p_samples)
        p_dim = len(p_samples[0]['c'])
        p_mat = np.empty((p_num, p_dim))
        for i, p_sample in enumerate(p_samples):
            p_mat[i, :] = p_sample['c']


        g_mat_list = self._eval_options['ufunc_ptr'](d_mat, p_mat)
        n_model_evals = d_num * p_num

        efp_values = [0.0]*d_num
        for i, g_mat in enumerate(g_mat_list):
            efp = 0.0
            for j, g_vec in enumerate(g_mat):
                if np.all(g_vec >= 0.0):
                    efp = round(efp + p_samples[j]['w'], ndigits=15)
                    if efp > 1.0:
                        print("EFP > 1.0:", efp, "j=", j)

            efp_values[i] = efp

        if not must_store_g:
            g_mat_list = []

        return efp_values, n_model_evals, g_mat_list

    def _evaluate_using_script(self, inputs):
        self._data_handler.set_inputs(inputs)
        self._data_handler.write_eval_data()

        if self.must_update_eval_script:
            self._script_handler.write_eval_script()
            self.must_update_eval_script = False

        the_script_fpne = self._script_handler.get_script_fpne()
        self._execute_script(the_script_fpne)
        self._data_handler.read_eval_data()
