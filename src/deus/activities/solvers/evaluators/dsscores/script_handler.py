from deus.activities.solvers.evaluators import EvaluationScriptHandler


class DSScoreEvalScriptHandler(EvaluationScriptHandler):
    def __init__(self, info, eval_method, eval_options, data_handler):
        super().__init__(info, eval_method)
        self.score_type = info['score_type']

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

    def _indicator_func_body(self, indent=""):
        atxt = indent + "score = 0.0\n"
        atxt += indent + "if np.all(g_vec >= 0.0):\n"
        atxt += indent + self._tab + "score = 1.0\n"
        return atxt

    def _sigmoid_func_body(self, indent=""):
        atxt = indent + "if np.all(g_vec[0, :] >= 0.0):\n"
        atxt += indent + self._tab + \
                "terms = [np.log(1.0 + np.exp(-g)) for g in g_vec[0, :]]\n"
        atxt += indent + "else:\n"
        atxt += indent + self._tab + "terms = []\n"
        atxt += indent + self._tab + "for g in g_vec[0, :]:\n"
        atxt += indent + self._2tabs + "if g >= 0.0:\n"
        atxt += indent + self._3tabs + "term = 0.693147180559945\n"
        atxt += indent + self._2tabs + "else:\n"
        atxt += indent + self._3tabs + "term = np.log(1.0 + np.exp(-g))\n"
        atxt += indent + self._2tabs + "terms.append(term)\n"
        atxt += indent + "score = -sum(terms)\n"
        return atxt

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

        atxt += indent + "g_dim = np.shape(g_list[0])[1]\n"

        atxt += indent + "score_values = np.ndarray(d_num)\n"
        atxt += indent + "for i, g_vec in enumerate(g_list):\n"

        if self.score_type == "indicator":
            atxt += indent + self._indicator_func_body(indent + self._tab)
        elif self.score_type == "sigmoid":
            atxt += indent + self._sigmoid_func_body(indent + self._tab)

        atxt += indent + self._tab + "score_values[i] = score"
        atxt += self._blank_line

        atxt += indent + "data['out'] = score_values\n"
        atxt += indent + "data['g_dim'] = g_dim\n"
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
            self._tab + "g_dim = np.shape(g_list[0])[1]\n" +\
            self._tab + "ochunk = []\n" +\
            self._tab + "for i, g_vec in enumerate(g_list):\n"

        if self.score_type == "indicator":
            atxt += self._indicator_func_body(self._2tabs)
        elif self.score_type == "sigmoid":
            atxt += self._sigmoid_func_body(self._2tabs)

        if must_store_g:
            atxt += self._2tabs + "item = {" \
                                  "'score': score, " \
                                  "'g_vec': g_vec, " \
                                  "'g_dim': g_dim}\n"
        else:
            atxt += self._2tabs + "item = {" \
                                  "'score': score, " \
                                  "'g_dim': g_dim}\n"

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
            self._tab + "data['out'] = [item['score'] for item in outputs]\n"+\
            self._tab + "data['g_dim'] = outputs[0]['g_dim']\n"

        if must_store_g:
            atxt += self._tab + "data['g_list'] = [item['g_vec'] "\
                               "for item in outputs]"
        else:
            atxt += self._tab + "data['g_list'] = []"

        return atxt