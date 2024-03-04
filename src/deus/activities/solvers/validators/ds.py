from deus.utils.assertions import DEUS_ASSERT


class ValidatorForDSSolver:
    @staticmethod
    def check_settings(settings):
        assert isinstance(settings, dict), \
            "The settings must be a dictionary."

        mkeys = ['score_evaluation', 'efp_evaluation', 'phases_setup']
        DEUS_ASSERT.has(mkeys, settings, "settings")

        spec_name = "score_evaluation"
        spec = settings[spec_name]
        assert isinstance(spec, dict), \
            "'score_evaluation' must be a dictionary."
        mkeys = ['method', 'score_type', 'store_constraints']
        DEUS_ASSERT.has(mkeys, spec, spec_name)
        if spec["method"] == "serial":
            mkeys = ['constraints_func_ptr']
            DEUS_ASSERT.has(mkeys, spec, spec_name, use_also=True)
        elif spec["method"] == "mppool":
            mkeys = ['pool_size']
            DEUS_ASSERT.has(mkeys, spec, spec_name)
            assert isinstance(spec['pool_size'], int), \
                "'n_processes' must be an integer."
            n_procs = spec['pool_size']
            assert (n_procs == -1 or n_procs >= 2), \
                "'pool_size' must be >=2 or -1.\n" \
                "-1: # processes = # logical cores."
        elif spec["method"] == "mpi":
            assert False, "Not implemented yet."
        else:
            assert False, "'score_evaluation' method not recognized."

        spec_name = "efp_evaluation"
        spec = settings[spec_name]
        assert isinstance(spec, dict), \
            "'efp_evaluation' must be a dictionary."
        if spec["method"] == "serial":
            mkeys = ['method', 'constraints_func_ptr', 'store_constraints']
            DEUS_ASSERT.has(mkeys, spec, spec_name)
        elif spec["method"] == "mppool":
            mkeys = ['method', 'pool_size', 'store_constraints']
            DEUS_ASSERT.has(mkeys, spec, spec_name)
            assert isinstance(spec['pool_size'], int), \
                "'n_processes' must be an integer."
            n_procs = spec['pool_size']
            assert (n_procs == -1 or n_procs >= 2), \
                "'pool_size' must be >=2 or -1.\n" \
                "-1: # processes = # logical cores."
        elif efp_eval["method"] == "mpi":
            assert False, "Not implemented yet."
        else:
            assert False, "'efp_evaluation' method not recognized."

        spec_name = "phases_setup"
        spec = settings[spec_name]
        assert isinstance(spec, dict), "'phases_setup' must be a dictionary."
        mkeys = ['initial', 'deterministic', 'nmvp_search', 'probabilistic']
        DEUS_ASSERT.has(mkeys, spec, spec_name)

        spec_name = "initial"
        spec = settings["phases_setup"][spec_name]
        mkeys = ['nlive', 'nproposals']
        DEUS_ASSERT.has(mkeys, spec, spec_name)

        spec_name = "deterministic"
        spec = settings["phases_setup"][spec_name]
        mkeys = ['skip']
        DEUS_ASSERT.has(mkeys, spec, spec_name)

        spec_name = "nmvp_search"
        spec = settings["phases_setup"][spec_name]
        mkeys = ['skip']
        DEUS_ASSERT.has(mkeys, spec, spec_name)
        if spec["skip"] is False:
            assert False, "Not implemented yet."

        spec_name = "probabilistic"
        spec = settings["phases_setup"][spec_name]
        mkeys = ['skip']
        DEUS_ASSERT.has(mkeys, spec, spec_name)
        if spec["skip"] is False:
            mkeys = ['nlive_change']
            DEUS_ASSERT.has(mkeys, spec, spec_name, use_also=True)

            spec_name = "nlive_change"
            spec = settings["phases_setup"]["probabilistic"][spec_name]
            mkeys = ['mode']
            DEUS_ASSERT.has(mkeys, spec, spec_name)

            if spec["mode"] == "user_given":
                mkeys = ['schedule']
                DEUS_ASSERT.has(mkeys, spec, spec_name, use_also=True)

                spec_name = "schedule"
                spec = \
                settings["phases_setup"]["probabilistic"]["nlive_change"][
                    spec_name]
                assert isinstance(spec, list), \
                    "'schedule' must be a list."
                for i, item in enumerate(spec):
                    assert isinstance(item, tuple), \
                        "'schedule' must contain (b, n, p) tuples, where:" \
                        "b - threshold reliability level (in [0, 1]); " \
                        "n - number of live points; " \
                        "p - number of proposals per iteration."
                    schedule_num_entries = len(spec)
                    if schedule_num_entries > 1 and i < schedule_num_entries - 1:
                        n1, n2 = spec[i][1], spec[i + 1][1]
                        assert (n1 < n2), \
                            "The number of live points must always increase."
            elif spec["mode"] == "auto":
                assert False, "Not implemented yet."
                pass
            else:
                assert False, "Invalid entry. Allowed values:\n " \
                              "'user_given', 'auto'."

    @staticmethod
    def check_algorithms(algos):
        assert isinstance(algos, dict), \
            "algorithms must be a dictionary."

        spec_name = "algorithms"
        spec = algos
        mkeys = ['sampling']
        DEUS_ASSERT.has(mkeys, spec, spec_name)
