from deus.activities.solvers.algorithms.composite import \
    NestedSamplingWithGlobalSearch


class CompositeAlgorithmFactory:
    @classmethod
    def create(cls, name, settings, algorithms):
        assert settings is not None, \
            "The settings must be defined."
        assert algorithms is not None, \
            "The algorithms must be defined."

        if name == "mc_sampling-ns_global":
            return NestedSamplingWithGlobalSearch(settings, algorithms)
        elif name == "mc_sampling-ns_local":
            pass  # TODO
            assert False, "Branch not done yet. :("
        elif name == "mc_sampling-mcmc":
            pass  # TODO
            assert False, "Branch not done yet. :("
