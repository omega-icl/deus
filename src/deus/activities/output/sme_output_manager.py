import pickle
from os.path import exists

from deus.activities.solvers.algorithms.points import XFPoint


class SetMembershipEstimationOutputManager:
    def __init__(self, cs_folder):
        assert exists(cs_folder)
        self.cs_folder = cs_folder
        self.output_filename = 'output'
        self.output_path_filename_extn = \
            self.cs_folder + self.output_filename + '.pkl'
        self.output = {
            "solution": {
                "initial_phase": {
                    "samples": {
                        "coordinates": [],
                        "phi": []
                    }
                },
                "search_phase": {
                    "samples": {
                        "coordinates": [],
                        "phi": []
                    },
                    "iterations": 1
                }
            },
            "performance": {
                "initial_phase": [],
                "search_phase": []
            }
        }

        self.alias = {
            "INITIAL": "initial_phase",
            "SEARCH": "search_phase"
        }

    def add(self, out_content):
        self.add_to_solution(out_content)
        self.add_to_performance(out_content)
        return None

    def add_to_solution(self, out_content):
        for container in out_content:
            src_phase = container["phase"]
            if container["samples"]:  # container has samples
                snk_phase = self.alias[src_phase]
                snk_root = self.output["solution"][snk_phase]

                coords = (XFPoint.coords_of(container["samples"])).tolist()
                snk_root["samples"]["coordinates"].extend(coords)

                fvalues = XFPoint.fvalues_of(container["samples"]).tolist()
                snk_root["samples"]["phi"].extend(fvalues)

    def add_to_performance(self, out_content):
        for container in out_content:
            src_phase = container["phase"]
            src_root = container["performance"]
            if src_phase == "INITIAL":
                element = {
                    "n_evals": {
                        "phi": src_root["n_evals"]["phi"],
                        "model": src_root["n_evals"]["model"]
                    },
                    "cpu_time": {
                        "uom": src_root["cpu_time"]["uom"],
                        "evaluating": src_root["cpu_time"]["evaluating"]
                    }
                }

            elif src_phase == "SEARCH":
                element = {
                    "iteration": container["iteration"],
                    "n_evals": {
                        "phi": src_root["n_evals"]["phi"],
                        "model": src_root["n_evals"]["model"]
                    },
                    "n_replacements_done": src_root["n_replacements_done"],
                    "cpu_time": {
                        "uom": src_root["cpu_time"]["uom"],
                        "proposing": src_root["cpu_time"]["proposing"],
                        "evaluating": src_root["cpu_time"]["evaluating"],
                        "iteration": src_root["cpu_time"]["iteration"]
                    }
                }

            snk_phase = self.alias[src_phase]
            self.output["performance"][snk_phase].append(element)

    def write_to_disk(self):
        with open(self.output_path_filename_extn, 'wb') as file:
            pickle.dump(self.output, file)

    def write_performance_summary(self, fmt="json"):
        fpn = self.cs_folder + "performance_summary"
        summary = self.__do_performance_summary()
        if fmt == "json":
            import json
            with open(fpn + "." + fmt, 'w') as file:
                json.dump(summary, file, indent=4, sort_keys=False)
        else:
            assert False, "Unrecognized 'fmt'."

    def __do_performance_summary(self):
        summary = {}
        for phase in self.output["performance"].keys():
            src = self.output["performance"][phase]
            if phase == "initial_phase":
                nme = sum([e["n_evals"]["model"] for e in src])
                cpu_t = sum([e["cpu_time"]["evaluating"] for e in src])
                element = {
                    "initial_phase": {
                        "n_model_evaluations": nme,
                        "cpu_s": cpu_t}
                }

            elif phase == "search_phase":
                nme = sum([e["n_evals"]["model"] for e in src])
                cpu_p = sum([e["cpu_time"]["proposing"] for e in src])
                cpu_e = sum([e["cpu_time"]["evaluating"] for e in src])
                cpu_t = sum([e["cpu_time"]["iteration"] for e in src])
                element = {
                    "search_phase": {
                        "n_model_evaluations": nme,
                        "cpu_s": {
                            "proposing": cpu_p,
                            "evaluating": cpu_e,
                            "total": cpu_t
                        }
                    }
                }

            else:
                assert False, "Unrecognized phase"

            summary.update(element)

        nme_overall = \
            summary["initial_phase"]["n_model_evaluations"] +\
            summary["search_phase"]["n_model_evaluations"]

        cpu_overall = \
            summary["initial_phase"]["cpu_s"] +\
            summary["search_phase"]["cpu_s"]["total"]

        element = {"overall": {
            "n_model_evaluations": nme_overall, "cpu_s": cpu_overall}}

        summary.update(element)
        return summary
