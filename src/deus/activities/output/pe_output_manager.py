import pickle
from os.path import exists

from deus.activities.solvers.algorithms.points import XFPoint


class ParameterEstimationOutputManager:
    def __init__(self, cs_folder):
        assert exists(cs_folder)
        self.cs_folder = cs_folder
        self.output_filename = 'output'
        self.output_path_filename_extn = self.cs_folder + \
                                      self.output_filename + '.pkl'
        self.output = {
            "solution": {
                "samples": {
                    "coordinates": [],
                    "log_l": [],
                    "weights": []
                },
                "nests": [],
                "log_z": {"hat": 0.0, "mean": 0.0, "sdev": 0.0},
                "post_prior_kldiv": 0.0
            },
            "performance": []
        }

    def add(self, out_content):
        self.add_to_solution(out_content)
        self.add_to_performance(out_content)

    def add_to_solution(self, out_content):
        for container in out_content:
            root = self.output["solution"]
            if container["samples"]:
                coords = (XFPoint.coords_of(container["samples"])).tolist()
                root["samples"]["coordinates"].extend(coords)
                fvalues = XFPoint.fvalues_of(container["samples"]).tolist()
                root["samples"]["log_l"].extend(fvalues)

            if container["nests"]:
                for nest in container["nests"]:
                    root["nests"].append(nest)

                root["log_z"]["hat"] = container["log_z"]["hat"]
                root["post_prior_kldiv"] = container["post_prior_kldiv"]

    def add_to_performance(self, out_content):
        for container in out_content:
            root = container["performance"]
            element = {
                "iteration": container["iteration"],
                "n_proposals": root["n_proposals"],
                "n_replacements": root["n_replacements"],
                "cpu_secs": {
                    "proposals": root["cpu_secs"]["proposals_generation"],
                    "lkhd_evals": root["cpu_secs"]["lkhd_evals"],
                    "total": root["cpu_secs"]["total"]
                }
            }
            self.output["performance"].append(element)

    def write_to_disk(self):
        with open(self.output_path_filename_extn, 'wb') as file:
            pickle.dump(self.output, file)

    def get_log_lw(self):
        root = self.output["solution"]["nests"]
        log_lw = [nest["log_lw"] for nest in root]
        return log_lw

    def get_logl_of_last(self, n):
        assert n > 0, "'n' must be >0."
        root = self.output["solution"]["samples"]["log_l"]
        log_l = root[-n:]
        return log_l

    def get_logl(self):
        root = self.output["solution"]["samples"]["log_l"]
        log_l = root
        return log_l

    def add_samples_weights(self, weights):
        root = self.output["solution"]["samples"]["weights"]
        root.extend(weights)

    def add_logz_statistics(self, mean, sdev):
        root = self.output["solution"]["log_z"]
        root["mean"] = mean
        root["sdev"] = sdev