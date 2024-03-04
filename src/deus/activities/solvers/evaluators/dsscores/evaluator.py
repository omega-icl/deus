import numpy as np

from deus.activities.solvers.evaluators import \
    ExpensiveFunctionEvaluator
from deus.activities.solvers.evaluators.dsscores.data_handler import\
    DSScoreEvalDataHandler
from deus.activities.solvers.evaluators.dsscores.script_handler import\
    DSScoreEvalScriptHandler


class DSScoreEvaluator(ExpensiveFunctionEvaluator):
    def __init__(self, info, eval_method, eval_options, p_best):
        super().__init__(info)
        vkeys = ['indicator', 'sigmoid']
        assert info['score_type'] in vkeys, "score_type is invalid."
        self.score_type = info['score_type']

        eval_path = self.ufunc_script_path
        self._data_handler = DSScoreEvalDataHandler(eval_path, p_best)

        script_handler_info = {
            'eval_path': self.ufunc_script_path,
            'ufunc_script_name': self.ufunc_script_name,
            'ufunc_name': self.ufunc_name,
            'score_type': self.score_type
        }
        self._script_handler = \
            DSScoreEvalScriptHandler(script_handler_info,
                                     eval_method,
                                     eval_options,
                                     self._data_handler)

        self._eval_method = None
        self.set_eval_method(eval_method)
        self._eval_options = None
        self.set_eval_options(eval_options)

        self.__g_dim = None

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
                if self._eval_options['store_constraints']:
                    g_list = the_data['g_list']
                else:
                    g_list = []

                if self.__g_dim is None:
                    self.__g_dim = the_data['g_dim']

            else:  # Don't use evaluation script
                func_values, g_list, _ = self._expensive_func(inputs)

        elif self._eval_method == "mppool":
            self._evaluate_using_script(inputs)
            the_data = self._data_handler.get_data()
            func_values = the_data['out']
            g_list = the_data['g_list']
            if self.__g_dim is None:
                self.__g_dim = the_data['g_dim']

        elif self._eval_method == "mpi":
            assert False, "not implemented yet."

        return func_values, g_list, self.__g_dim

    def _expensive_func(self, inputs):
        '''
        :param inputs: array MxD, each row is a point in D-space
        :return:
            * score_values: 1d array of M scalars that are score(s);
            * g_list (optional): list of 2d arrays representing
            g(d, p_best)
        '''
        d_mat = inputs
        p_best = np.array([self._data_handler.get_p_best()])
        must_store_g = self._eval_options['store_constraints']

        d_shape = np.shape(d_mat)
        if len(d_shape) == 1:
            d_num, d_dim = 1, d_shape
        else:
            d_num, d_dim = d_shape
        score_values = np.ndarray(d_num)

        g_list = self._eval_options['ufunc_ptr'](d_mat, p_best)

        if self.__g_dim is None:
            self.__g_dim = np.shape(g_list[0])[1]

        for i, g_vec in enumerate(g_list):
            if self.score_type == "indicator":
                score = 0.0
                if np.all(g_vec[0, :] >= 0.0):
                    score = 1.0

            elif self.score_type == "sigmoid":
                if np.all(g_vec[0, :] >= 0.0):
                    terms = [np.log(1.0 + np.exp(-g)) for g in g_vec[0, :]]
                else:
                    terms = []
                    for g in g_vec[0, :]:
                        if g >= 0.0:
                            term = 0.693147180559945  # np.log(2)
                        else:
                            term = np.log(1.0 + np.exp(-g))
                        terms.append(term)
                score = -sum(terms)

            else:
                assert False, "unrecognized score_type."

            score_values[i] = score

        if not must_store_g:
            g_list = []

        return score_values, g_list, self.__g_dim

    def _evaluate_using_script(self, inputs):
        self._data_handler.set_inputs(inputs)
        self._data_handler.write_eval_data()

        if self.must_update_eval_script:
            self._script_handler.write_eval_script()
            self.must_update_eval_script = False

        the_script_fpne = self._script_handler.get_script_fpne()
        self._execute_script(the_script_fpne)
        self._data_handler.read_eval_data()
