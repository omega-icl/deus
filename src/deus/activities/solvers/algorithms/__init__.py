import abc

from deus.activities.solvers.algorithms.points import BayesPoint, XFPoint


class Algorithm:
    def __init__(self):
        pass

    @abc.abstractmethod
    def get_type(self):
        raise NotImplementedError

    @abc.abstractmethod
    def get_ui_name(self):
        raise NotImplementedError


class PrimitiveAlgorithm(Algorithm):
    def __init__(self):
        super().__init__()


class CompositeAlgorithm(Algorithm):
    def __init__(self):
        super().__init__()
        self.settings = {}
        self.algorithms = []
