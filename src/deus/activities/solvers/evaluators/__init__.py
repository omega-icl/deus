import abc
from pathlib import Path
import pickle
from subprocess import call


class ExpensiveFunctionEvaluator:
    def __init__(self, info):
        assert isinstance(info, dict), "'info' must be a dict."
        self.ufunc_script_path = info['ufunc_script_path']
        self.ufunc_script_name = info['ufunc_script_name']
        self.ufunc_name = info['ufunc_name']

        # behavior
        self.must_update_eval_script = True

    @abc.abstractmethod
    def set_eval_method(self, eval_method):
        raise NotImplementedError

    @abc.abstractmethod
    def set_eval_options(self, eval_options):
        raise NotImplementedError

    @abc.abstractmethod
    def evaluate(self, inputs):
        raise NotImplementedError

    @abc.abstractmethod
    def _expensive_func(self, inputs):
        raise NotImplementedError

    def _execute_script(self, script_fpne):
        call(["python", script_fpne])


class EvaluationDataHandler:
    def __init__(self, eval_path):
        self.eval_folder_path = Path(eval_path)

        self._data = {'in': [], 'out': []}

        self._eval_data_fn = "temp_evaluation_data"
        self._eval_data_fne = self._eval_data_fn + ".pkl"
        self._eval_data_fpne = self.eval_folder_path / self._eval_data_fne

    def set_inputs(self, inputs):
        self._data['in'] = inputs
        self._data['out'] = []

    def get_data(self):
        return self._data

    def get_data_fne(self):
        return self._eval_data_fne

    def write_eval_data(self):
        with open(self._eval_data_fpne, 'wb') as file:
            pickle.dump(self._data, file)

    def read_eval_data(self):
        with open(self._eval_data_fpne, 'rb') as file:
            self._data = pickle.load(file)


class EvaluationScriptHandler:
    def __init__(self, info, eval_method):
        self.eval_folder_path = Path(info['eval_path'])
        self.ufunc_script_name = info['ufunc_script_name']
        self.ufunc_name = info['ufunc_name']
        self.eval_method = eval_method

        self._ufunc_script_fne = self.ufunc_script_name + '.py'
        self._eval_script_fn = "temp_evaluation_script"
        self._eval_script_fne = self._eval_script_fn + ".py"
        self._eval_script_fpne = self.eval_folder_path / self._eval_script_fne

        self._tab = "    "
        self._2tabs = self._tab + self._tab
        self._3tabs = self._2tabs + self._tab
        self._blank_line = "\n\n"
        self._2blank_lines = "\n\n\n"

    def get_script_fpne(self):
        return str(self._eval_script_fpne)

    def write_eval_script(self):
        script = self._evaluation_script()
        with open(self._eval_script_fpne, 'w') as file:
            file.write(script)

    @abc.abstractmethod
    def _evaluation_script(self):
        raise NotImplementedError

    def _eval_script_header(self):
        atxt = ""
        atxt += \
            "from pathlib import Path\n"\
            "import pickle" +\
            self._blank_line +\
            "from " + self.ufunc_script_name + " import " + self.ufunc_name
        return atxt

    def _data_pickle_reading(self, data_fne, indent=""):
        atxt = \
            indent + "data_path = Path.cwd()\n" +\
            indent + "data_fpne = data_path / \"" + data_fne + "\" \n" +\
            indent + "with open(data_fpne, 'rb') as file:\n" +\
            indent + self._tab + "data = pickle.load(file)"
        return atxt

    def _data_pickle_writing(self, indent=""):
        atxt = \
            indent + "with open(data_fpne, 'wb') as file:\n" +\
            indent + self._tab + "pickle.dump(data, file)"
        return atxt

    def _guard(self):
        atxt = "if __name__ == \"__main__\":"
        return atxt
