import pickle
from os.path import exists


class OutputManager:
    def __init__(self, cs_folder):
        assert exists(cs_folder)
        self.cs_folder = cs_folder
        self.results_filename = 'results'
        self.results_path_filename_extn = self.cs_folder + \
                                      self.results_filename + '.pkl'

    def write_results(self, results):
        assert isinstance(results, dict), \
            "'results' must be a dictionary."

        if exists(self.results_path_filename_extn):
            results_buffer = self.read_results()
            for k, v in results.items():
                if isinstance(v, list):
                    results_buffer[k].extend(v)
                elif isinstance(v, dict):
                    results_buffer[k].update(v)
                else:
                    results_buffer[k] = v
        else:
            results_buffer = results

        with open(self.results_path_filename_extn, 'wb') as file:
            pickle.dump(results_buffer, file)

    def read_results(self):
        assert exists(self.results_path_filename_extn),\
            self.results_filename + ".pkl does not exist."
        with open(self.results_path_filename_extn, 'rb') as file:
            results_buffer = pickle.load(file)
        return results_buffer
