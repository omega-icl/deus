from deus.activities.solvers.evaluators import EvaluationScriptHandler


class SMEScoreEvalScriptHandler(EvaluationScriptHandler):
    def __init__(self, info, eval_method, eval_options, data_handler):
        super().__init__(info, eval_method)

        self.eval_options = eval_options
        self.data_handler = data_handler

    def _evaluation_script(self):
        if self.eval_method == "serial":
            script = self._serial_script()

        elif self.eval_method == "mppool":
            script = self._mppool_script()

        elif self.eval_method == "mpi":
            assert False, "not implemented yet."

        return script

    def _score_func_body(self, indent=""):
        atxt = indent + "normalization_vec = e_s2b_ratio/e_bounds\n"
        atxt += indent + "e_mat_bounds = np.array([e_bounds]*p_num)\n"
        atxt += indent + "e_mat = " + self.ufunc_name + "(p_mat)\n"
        atxt += indent + "abs_mat = np.abs(e_mat)\n"
        atxt += indent + "diff_mat = abs_mat - e_mat_bounds\n"
        atxt += indent + "max_mat = np.maximum(diff_mat, 0.0)\n"
        atxt += indent + "prod_mat = normalization_vec*max_mat\n"
        atxt += indent + "final_mat = prod_mat*prod_mat\n"
        atxt += indent + "score_values = -0.5*np.sum(final_mat, axis=1)\n"
        atxt += indent + "nans = np.isnan(score_values)\n"
        atxt += indent + "for i, is_a_nan in enumerate(nans):\n"
        atxt += indent + self._tab + "if is_a_nan:\n"
        atxt += indent + self._2tabs + "score_values[i] = -np.inf"
        return atxt

    # Serial Evaluation
    def _serial_script(self):
        script = self._serial_script_header()
        script += self._2blank_lines

        data_fne = self.data_handler.get_data_fne()
        script += super()._data_pickle_reading(data_fne, indent="")
        script += self._2blank_lines

        script += self._serial_smescore_evaluation(indent="")
        script += self._2blank_lines

        script += super()._data_pickle_writing(indent="") + "\n"
        return script

    def _serial_script_header(self):
        atxt = "import numpy as np\n"
        atxt += super()._eval_script_header()
        return atxt

    def _serial_smescore_evaluation(self, indent=""):
        atxt = indent + "p_mat = data['in']\n"
        atxt += indent + "e_bounds = np.array(data['e_bounds'])\n"
        atxt += indent + "e_s2b_ratio = data['s2e_ratio']"\
                + self._blank_line

        atxt += indent + "p_shape = np.shape(p_mat)\n"
        atxt += indent + "if len(p_shape) == 1:\n"
        atxt += indent + self._tab + "p_num, p_dim = 1, p_shape\n"
        atxt += indent + "else:\n"
        atxt += indent + self._tab + "p_num, p_dim = p_shape" + \
                self._blank_line

        atxt += indent + self._score_func_body(indent) \
                + self._blank_line

        atxt += indent + "data['out'] = score_values"
        return atxt

    # Multiprocess Pool Evaluation
    def _mppool_script(self):
        script = self._mppool_script_header()
        script += self._2blank_lines

        data_fne = self.data_handler.get_data_fne()
        script += super()._data_pickle_reading(data_fne, indent="")
        script += self._2blank_lines

        script += self._mppool_global_data()
        script += self._2blank_lines

        script += self._mppool_chunks_evaluation_func()
        script += self._2blank_lines

        script += self._guard()
        script += self._blank_line

        script += self._mppool_chunks_and_pool_creation(indent=self._tab)
        script += self._blank_line

        script += self._mpool_chunks_collection()
        script += self._blank_line

        script += super()._data_pickle_writing(indent=self._tab) + "\n"
        return script

    def _mppool_script_header(self):
        atxt = "from multiprocessing import Pool, cpu_count\n"
        atxt += "import numpy as np\n"
        atxt += super()._eval_script_header()
        return atxt

    def _mppool_global_data(self):
        atxt = "e_bounds = np.array(data['e_bounds'])\n"
        atxt += "e_s2b_ratio = data['s2e_ratio']"
        return atxt

    def _mppool_chunks_evaluation_func(self):
        atxt = "def calculate_output_for(ichunk):\n"
        atxt += \
            self._tab + "p_mat = ichunk" +\
            self._blank_line

        atxt += self._tab + "p_shape = np.shape(p_mat)\n"
        atxt += self._tab + "if len(p_shape) == 1:\n"
        atxt += self._2tabs + "p_num, p_dim = 1, p_shape\n"
        atxt += self._tab + "else:\n"
        atxt += self._2tabs + "p_num, p_dim = p_shape" + \
                self._blank_line

        atxt +=\
            self._tab + "normalization_vec = e_s2b_ratio/e_bounds\n" +\
            self._tab + "e_mat_bounds = np.array([e_bounds]*p_num)\n" +\
            self._tab + "e_mat = " + self.ufunc_name + "(p_mat)\n" +\
            self._tab + "abs_mat = np.abs(e_mat)\n" +\
            self._tab + "diff_mat = abs_mat - e_mat_bounds\n" +\
            self._tab + "max_mat = np.maximum(diff_mat, 0.0)\n" +\
            self._tab + "prod_mat = normalization_vec*max_mat\n" +\
            self._tab + "final_mat = prod_mat*prod_mat\n" +\
            self._tab + "score_values = -0.5*np.sum(final_mat, axis=1)\n" + \
            self._tab + "nans = np.isnan(score_values)\n" + \
            self._tab + "for i, is_a_nan in enumerate(nans):\n" + \
            self._2tabs + "if is_a_nan:\n" + \
            self._3tabs + "score_values[i] = -np.inf\n" + \
            self._tab + "ochunk = [score_values]\n" +\
            self._tab + "return ochunk"
        return atxt

    def _mppool_chunks_and_pool_creation(self, indent=""):
        n_pool_processes = self.eval_options['pool_size']

        atxt = indent + "inputs = data['in']\n"
        if n_pool_processes == -1:
            pool_size = "cpu_count()"
        else:
            pool_size = str(n_pool_processes)
        atxt += indent + "n_processes = " + pool_size + "\n"

        atxt +=\
            indent + "n_inputs = int(len(inputs))\n" +\
            indent + "chunk_size = int(n_inputs/n_processes)\n" +\
            indent + "input_chunks = [inputs[i*chunk_size:(i+1)*chunk_size] " \
                     "for i in range(n_processes-1)]\n" +\
            indent + "input_chunks.append(inputs[(n_processes-1)*chunk_size:])\n" +\
            indent + "with Pool(n_processes) as the_pool:\n" + \
            indent + self._tab + "output_chunks = the_pool.map(" \
                                 "calculate_output_for, input_chunks)"
        return atxt

    def _mpool_chunks_collection(self):
        atxt = \
            self._tab + "outputs = []\n" +\
            self._tab + "for chunk in output_chunks:\n" +\
            self._2tabs + "outputs.extend(chunk[0])" +\
            self._blank_line + \
            self._tab + "data['out'] = outputs"
        return atxt
