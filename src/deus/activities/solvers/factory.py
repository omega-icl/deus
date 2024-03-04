from deus.activities.solvers.ds_ns_solver import \
    DesignSpaceSolverUsingNS
from deus.activities.solvers.pe_ns_solver import \
    ParameterEstimationSolverUsingNS
from deus.activities.solvers.sme_ns_solver import \
    SetMembershipEstimationSolverUsingNS


class SolverFactory:
    @classmethod
    def create(cls, name, cs_path, problem, settings, algorithms):
        assert name is not None, \
            "The solver must be defined."
        assert problem is not None, \
            "The problem must be defined."
        assert settings is not None, \
            "The settings must be defined."
        assert algorithms is not None, \
            "The algorithms must be defined."

        if name == "pe-ns":
            return ParameterEstimationSolverUsingNS(cs_path,
                                                    problem,
                                                    settings,
                                                    algorithms)

        elif name == "dsc-ns":
            return DesignSpaceSolverUsingNS(cs_path,
                                            problem,
                                            settings,
                                            algorithms)

        elif name == "sme-ns":
            return SetMembershipEstimationSolverUsingNS(cs_path,
                                                        problem,
                                                        settings,
                                                        algorithms)

        else:
            assert False, "Solver not found."