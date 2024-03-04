import pickle
from os.path import exists

from deus.activities.solvers.algorithms.points import XFPoint


class DesignSpaceOutputManager:
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
                    },
                    "constraints_info": []
                },
                "deterministic_phase": {
                    "samples": {
                        "coordinates": [],
                        "phi": []
                    },
                    "constraints_info": [],
                    "iterations": 1
                },
                "transition_phase": {
                    "samples": {
                        "coordinates": [],
                        "phi": []
                    },
                    "constraints_info": []
                },
                "probabilistic_phase": {
                    "samples": {
                        "coordinates": [],
                        "phi": []
                    },
                    "constraints_info": []
                }
            },
            "performance": {
                "initial_phase": [],
                "deterministic_phase": [],
                "transition_phase": [],
                "probabilistic_phase": []
            }
        }

        self.alias = {
            "INITIAL": "initial_phase",
            "DETERMINISTIC": "deterministic_phase",
            "NMVP_SEARCH": "nmvp_search_phase",
            "TRANSITION": "transition_phase",
            "PROBABILISTIC": "probabilistic_phase"
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

                g_info = container["constraints_info"]
                snk_root["constraints_info"].extend(g_info)

    def add_to_performance(self, out_content):
        for container in out_content:
            src_phase = container["phase"]
            src_root = container["performance"]
            if src_phase in ["INITIAL", "TRANSITION"]:
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

            elif src_phase == "NMVP_SEARCH":
                assert False, "Not implemented yet."

            elif src_phase == "DETERMINISTIC":
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

            elif src_phase == "PROBABILISTIC":
                element = {
                    "iteration": container["iteration"],
                    "n_evals": {
                        "phi": {
                            "main": src_root["n_evals"]["phi"]["main"],
                            "topup": src_root["n_evals"]["phi"]["topup"]
                        },
                        "model": {
                            "main": src_root["n_evals"]["model"]["main"],
                            "topup": src_root["n_evals"]["model"]["topup"]
                        },
                    },
                    "n_replacements_done": src_root["n_replacements_done"],
                    "cpu_time": {
                        "uom": src_root["cpu_time"]["uom"],
                        "proposing": {
                            "main": src_root["cpu_time"]["proposing"]["main"],
                            "topup": src_root["cpu_time"]["proposing"]["topup"]
                        },
                        "evaluating": {
                            "main": src_root["cpu_time"]["evaluating"]["main"],
                            "topup": src_root["cpu_time"]["evaluating"]["topup"]
                        },
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

            elif phase == "deterministic_phase":
                nme = sum([e["n_evals"]["model"] for e in src])
                cpu_p = sum([e["cpu_time"]["proposing"] for e in src])
                cpu_e = sum([e["cpu_time"]["evaluating"] for e in src])
                cpu_t = sum([e["cpu_time"]["iteration"] for e in src])
                element = {
                    "deterministic_phase": {
                        "n_model_evaluations": nme,
                        "cpu_s": {
                            "proposing": cpu_p,
                            "evaluating": cpu_e,
                            "total": cpu_t
                        }
                    }
                }

            elif phase == "transition_phase":
                nme = sum([e["n_evals"]["model"] for e in src])
                cpu_t = sum([e["cpu_time"]["evaluating"] for e in src])
                element = {
                    "transition_phase": {
                        "n_model_evaluations": nme,
                        "cpu_s": cpu_t}
                }

            elif phase == "probabilistic_phase":
                nme_main = sum([e["n_evals"]["model"]["main"] for e in src])
                cpu_p_main = sum([e["cpu_time"]["proposing"]["main"]
                                  for e in src])
                cpu_e_main = sum([e["cpu_time"]["evaluating"]["main"]
                                  for e in src])

                nme_topup = sum([e["n_evals"]["model"]["topup"] for e in src])
                cpu_p_topup = sum([e["cpu_time"]["proposing"]["topup"]
                                   for e in src])
                cpu_e_topup = sum([e["cpu_time"]["evaluating"]["topup"]
                                   for e in src])

                cpu_t = sum([e["cpu_time"]["iteration"] for e in src])

                element = {
                    "probabilistic_phase": {
                        "n_model_evaluations": {
                            "main": nme_main,
                            "topup": nme_topup,
                            "total": nme_main + nme_topup
                        },
                        "cpu_s": {
                            "main": {
                                "proposing": cpu_p_main,
                                "evaluating": cpu_e_main,
                                "total": cpu_p_main + cpu_e_main
                            },
                            "topup": {
                                "proposing": cpu_p_topup,
                                "evaluating": cpu_e_topup,
                                "total": cpu_p_topup + cpu_e_topup
                            },
                            "total": cpu_t
                        }
                    }
                }
            else:
                assert False, "Unrecognized phase"

            summary.update(element)

        nme_overall = \
            summary["initial_phase"]["n_model_evaluations"] +\
            summary["deterministic_phase"]["n_model_evaluations"] +\
            summary["transition_phase"]["n_model_evaluations"] +\
            summary["probabilistic_phase"]["n_model_evaluations"]["main"] +\
            summary["probabilistic_phase"]["n_model_evaluations"]["topup"]

        cpu_overall = \
            summary["initial_phase"]["cpu_s"] +\
            summary["deterministic_phase"]["cpu_s"]["total"] +\
            summary["transition_phase"]["cpu_s"] + \
            summary["probabilistic_phase"]["cpu_s"]["total"]

        element = {"overall": {
            "n_model_evaluations": nme_overall, "cpu_s": cpu_overall}}

        summary.update(element)
        return summary
