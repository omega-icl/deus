from deus import utils
from deus.utils.assertions import DEUS_ASSERT


class ValidatorForPESolver:
    @staticmethod
    def check_settings(settings):
        assert isinstance(settings, dict), \
            "The settings must be a dictionary."

        spec = settings
        spec_name = "settings"
        mkeys = ['log_lkhd_evaluation', 'stop_criteria']
        DEUS_ASSERT.has(mkeys, settings, spec_name)

        spec_name = "log_lkhd_evaluation"
        spec = settings[spec_name]
        assert isinstance(spec, dict),\
            "'log_lkhd_evaluation' must be a dictionary."
        DEUS_ASSERT.has(['method'], spec, spec_name)

        if spec["method"] == "serial":
            mkeys = ['log_lkhd_ptr']
            DEUS_ASSERT.has(mkeys, spec, spec_name, use_also=True)

        elif spec["method"] == "mppool":
            mkeys = ['pool_size']
            DEUS_ASSERT.has(mkeys, spec, spec_name, use_also=True)
            assert isinstance(spec['pool_size'], int), \
                "'n_processes' must be an integer."
            n_procs = spec['pool_size']
            assert (n_procs == -1 or n_procs >= 2), \
                "'pool_size' must be >=2 or -1.\n"\
                "-1: # processes = # logical cores."

        elif spec["method"] == "mpi":
            assert False, "Not implemented yet."
        else:
            assert False, "'log_lkhd_evaluation' method not recognized."

        specified_stop_criteria = utils.keys_in(settings["stop_criteria"])
        assert 'contribution_to_evidence' in specified_stop_criteria, \
            "'contribution_to_evidence' must be the stop criterion."

    @staticmethod
    def check_algorithms(algos):
        assert isinstance(algos, dict), \
            "algorithms must be a dictionary."

        spec = algos
        spec_name = "algorithms"
        mkeys = ['sampling']
        DEUS_ASSERT.has(mkeys, spec, spec_name)
