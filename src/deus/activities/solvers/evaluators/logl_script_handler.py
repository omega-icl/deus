from deus.activities.solvers.evaluators import EvaluationScriptHandler


class LogLkhdEvalScriptHandler(EvaluationScriptHandler):
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
        script += self._blank_line

        script += self._serial_logl_evaluation(indent="")
        script += self._blank_line

        script += super()._data_pickle_writing(indent="") + "\n"
        return script

    def _serial_script_header(self):
        atxt = "import numpy as np\n"
        atxt += super()._eval_script_header()
        return atxt

    def _serial_logl_evaluation(self, indent=""):
        atxt = ""
        atxt += \
            indent + "p_mat = data['in']\n" +\
            indent + "assert len(np.shape(p_mat)) == 2, " \
                     "\"'p_mat' must be a 2d array.\"\n" +\
            indent + "logl_values = " + self.ufunc_name + "(p_mat)" +\
            self._blank_line +\
            indent + "data['out'] = logl_values"
        return atxt

    # Multiprocess Pool Evaluation
    def _mppool_script(self):
        script = self._mppool_script_header()
        script += self._2blank_lines

        script += self._mppool_chunks_evaluation_func()
        script += self._2blank_lines

        script += self._guard()
        script += self._blank_line

        data_fne = self.data_handler.get_data_fne()
        script += super()._data_pickle_reading(data_fne, indent=self._tab)
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

    def _mppool_chunks_evaluation_func(self):
        atxt = "def calculate_output_for(ichunk):\n"
        atxt += \
            self._tab + "assert len(np.shape(ichunk)) == 2, " \
                        "\"'ichunk' must be a 2d array.\"\n" + \
            self._tab + "ochunk = " + self.ufunc_name + "(ichunk)\n" +\
            self._tab + "return ochunk"
        # self._tab + "ochunk = []\n" +\
            # self._tab + "ochunk = " + self.ufunc_name +"(ichunk)\n" +\
            # self._tab + "ochunk.append(logl_values)\n" +\
        return atxt

    def _mppool_chunks_and_pool_creation(self, indent=""):
        atxt = indent + "inputs = data['in']\n"

        n_pool_processes = self.eval_options['pool_size']
        if n_pool_processes == -1:
            pool_size = "cpu_count()"
        else:
            pool_size = str(n_pool_processes)

        atxt +=\
            indent + "n_processes = " + pool_size + "\n" +\
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
        atxt = self._tab + "data['out'] = np.concatenate(output_chunks)"
        return atxt
