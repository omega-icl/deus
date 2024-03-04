from deus.activities.solvers.evaluators import EvaluationScriptHandler


class EFPEvalScriptHandler(EvaluationScriptHandler):
    def __init__(self, info, eval_method, eval_options, data_handler):
        super().__init__(info, eval_method)
        self.eval_options = eval_options
        self.data_handler = data_handler

        self._4tabs = self._3tabs + self._tab

    def _evaluation_script(self):
        if self.eval_method == "serial":
            script = self._serial_script()

        elif self.eval_method == "mppool":
            # assert False, "Not implemented yet."
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

        script += self._serial_efp_evaluation(indent="")
        script += self._2blank_lines

        script += super()._data_pickle_writing(indent="") + "\n"
        return script

    def _serial_script_header(self):
        atxt = "import numpy as np\n"
        atxt += super()._eval_script_header()
        return atxt

    def _serial_efp_evaluation(self, indent=""):
        must_store_g = self.eval_options['store_constraints']
        # must_accelerate = self.eval_options['acceleration']

        atxt = indent + "d_mat = data['in']\n"
        atxt += indent + "p_samples = data['p_samples']\n"
        # atxt += indent + "worst_efp = data['worst_efp']\n" \
        #         + self._blank_line
        atxt += self._blank_line

        atxt += indent + "d_shape = np.shape(d_mat)\n"
        atxt += indent + "if len(d_shape) == 1:\n"
        atxt += indent + self._tab + "d_num, d_dim = 1, d_shape\n"
        atxt += indent + "else:\n"
        atxt += indent + self._tab + "d_num, d_dim = d_shape\n" \
                + self._blank_line

        atxt += indent + "p_num = len(p_samples)\n"
        atxt += indent + "p_dim = len(p_samples[0]['c'])\n"
        atxt += indent + "p_mat = np.empty((p_num, p_dim))\n"
        atxt += indent + "for i, p_sample in enumerate(p_samples):\n"
        atxt += indent + self._tab + "p_mat[i, :] = p_sample['c']\n"\
                + self._blank_line

        atxt += indent + "g_mat_list = " + self.ufunc_name + "(d_mat, p_mat)\n"
        atxt += indent + "n_model_evals = d_num * p_num\n"
        atxt += indent + "efp_values = [0.0]*d_num\n"

        atxt += indent + "for i, g_mat in enumerate(g_mat_list):\n"
        atxt += indent + self._tab + "efp = 0.0\n"
        atxt += indent + self._tab + "for j, g_vec in enumerate(g_mat):\n"
        atxt += indent + self._2tabs + "if np.all(g_vec >= 0.0):\n"
        atxt += indent + self._3tabs + \
                "efp = round(efp + p_samples[j]['w'], ndigits=15)\n"
        atxt += indent + self._tab + "efp_values[i] = efp\n"

        atxt += indent + "data['out'] = efp_values\n"
        atxt += indent + "data['n_model_evals'] = n_model_evals\n"
# TODO g storage
        if must_store_g:
            atxt += indent + "data['g_list'] = g_mat_list"
        return atxt

    # Multiprocessing Pool Evaluation
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
        atxt = "p_samples = data['p_samples']"
        # atxt += "worst_efp = data['worst_efp']"
        return atxt

    def _mppool_chunks_evaluation_func(self):
        must_store_g = self.eval_options['store_constraints']
        # must_accelerate = self.eval_options['acceleration']

        atxt = "def calculate_output_for(ichunk):\n"

        atxt += self._tab + "p_num = len(p_samples)\n"
        atxt += self._tab + "p_dim = len(p_samples[0]['c'])\n"
        atxt += self._tab + "p_mat = np.empty((p_num, p_dim))\n"
        atxt += self._tab + "for i, p_sample in enumerate(p_samples):\n"
        atxt += self._2tabs + "p_mat[i, :] = p_sample['c']" + self._blank_line

        atxt += self._tab + "d_shape = np.shape(ichunk)\n"
        atxt += self._tab + "if len(d_shape) == 1:\n"
        atxt += self._2tabs + "d_num, d_dim = 1, d_shape\n"
        atxt += self._tab + "else:\n"
        atxt += self._2tabs + "d_num, d_dim = d_shape" + self._blank_line

        atxt +=\
            self._tab + "n_model_evals = p_num\n" +\
            self._tab + "g_mat_list = " + self.ufunc_name + \
            "(ichunk, p_mat)\n" +\
            self._tab + "ochunk = []\n" +\
            self._tab + "for i, g_mat in enumerate(g_mat_list):\n" +\
            self._2tabs + "efp = 0.0\n" +\
            self._2tabs + "for j, g_vec in enumerate(g_mat):\n" +\
            self._3tabs + "if np.all(g_vec >= 0.0):\n" + \
            self._4tabs + "efp = round(efp + p_samples[j]['w'], ndigits=15)\n"

        if must_store_g:
            atxt += self._2tabs + \
                    "item = {'efp': efp, " \
                    "'nme': n_model_evals, " \
                    "'g_list': g_mat}\n"
        else:
            atxt += self._2tabs + "item = {'efp': efp, 'nme': n_model_evals}\n"

        atxt += self._2tabs + "ochunk.append(item)\n" +\
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
            self._tab + "data['out'] = [item['efp'] for item in outputs]\n" +\
            self._tab + "data['n_model_evals'] = [item['nme'] for item in" \
                        " outputs]\n"
        if must_store_g:
            atxt += self._tab + "data['g_list'] = [item['g_list'] "\
                               "for item in outputs]"
        else:
            atxt += self._tab + "data['g_list'] = []"

        return atxt
