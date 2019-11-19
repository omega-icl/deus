from deus.activities.solvers.evaluators import EvaluationScriptHandler


class DSScoreEvalScriptHandler(EvaluationScriptHandler):
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

    # Serial Evaluation
    def _serial_script(self):
        script = self._serial_script_header()
        script += self._2blank_lines

        data_fne = self.data_handler.get_data_fne()
        script += super()._data_pickle_reading(data_fne, indent="")
        script += self._2blank_lines

        script += self._serial_dsscore_evaluation(indent="")
        script += self._2blank_lines

        script += super()._data_pickle_writing(indent="") + "\n"
        return script

    def _serial_script_header(self):
        atxt = "import numpy as np\n"
        atxt += super()._eval_script_header()
        return atxt

    def _serial_dsscore_evaluation(self, indent=""):
        must_store_g = self.eval_options['store_constraints']

        atxt = indent + "d_mat = data['in']\n"
        atxt += indent + "p_best = np.array([data['p_best']])" \
                + self._blank_line

        atxt += indent + "d_shape = np.shape(d_mat)\n"
        atxt += indent + "if len(d_shape) == 1:\n"
        atxt += indent + self._tab + "d_num, d_dim = 1, d_shape\n"
        atxt += indent + "else:\n"
        atxt += indent + self._tab + "d_num, d_dim = d_shape\n"

        atxt += indent + "g_list = " + self.ufunc_name + "(d_mat, p_best)\n"

        atxt += indent + "score_values = np.ndarray(d_num)\n"
        atxt += indent + "for i, g_vec in enumerate(g_list):\n"
        atxt += indent + self._tab + "score = 0.0\n"
        atxt += indent + self._tab + "if np.all(g_vec >= 0.0):\n"
        atxt += indent + self._2tabs + "score = 1.0\n"
        atxt += indent + self._tab + "score_values[i] = score" +\
                self._blank_line

        atxt += indent + "data['out'] = score_values\n"
        if must_store_g:
            atxt += indent + "data['g_list'] = g_list"

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
        script += self._2blank_lines

        script += self._mpool_chunks_collection()
        script += self._2blank_lines

        script += super()._data_pickle_writing(indent=self._tab) + "\n"
        return script

    def _mppool_script_header(self):
        atxt = "from multiprocessing import Pool, cpu_count\n"
        atxt += "import numpy as np\n"
        atxt += super()._eval_script_header()
        return atxt

    def _mppool_global_data(self):
        atxt = "p_best = np.array([data['p_best']])"
        return atxt

    def _mppool_chunks_evaluation_func(self):
        must_store_g = self.eval_options['store_constraints']

        atxt = "def calculate_output_for(ichunk):\n"
        atxt += \
            self._tab + "g_list = " + self.ufunc_name + "(ichunk, p_best)\n" +\
            self._tab + "ochunk = []\n" +\
            self._tab + "for i, g_vec in enumerate(g_list):\n" +\
            self._2tabs + "score = 0.0\n" +\
            self._2tabs + "if np.all(g_vec >= 0.0):\n" +\
            self._3tabs + "score = 1.0\n"

        if must_store_g:
            atxt += self._2tabs + "item = {'score': score, 'g_vec': g_vec}\n"
        else:
            atxt += self._2tabs + "item = {'score': score}\n"

        atxt += self._2tabs + "ochunk.append(item)\n" + \
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
        must_store_g = self.eval_options['store_constraints']

        atxt = \
            self._tab + "outputs = []\n" +\
            self._tab + "for chunk in output_chunks:\n" +\
            self._2tabs + "outputs.extend(chunk)" +\
            self._blank_line + \
            self._tab + "data['out'] = [item['score'] for item in outputs]\n"

        if must_store_g:
            atxt += self._tab + "data['g_list'] = [item['g_vec'] "\
                               "for item in outputs]"
        else:
            atxt += self._tab + "data['g_list'] = []"

        return atxt