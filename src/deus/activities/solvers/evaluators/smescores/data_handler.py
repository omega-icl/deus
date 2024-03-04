from deus.activities.solvers.evaluators import EvaluationDataHandler


class SMEScoreEvalDataHandler(EvaluationDataHandler):
    def __init__(self, eval_path, e_bounds, s2e_ratio):
        super().__init__(eval_path)
        self._e_bounds = None
        self.set_e_bounds(e_bounds)
        self._s2e_ratio = None
        self.set_spread_to_error_bound(s2e_ratio)

    def set_e_bounds(self, e_bounds):
        self._e_bounds = e_bounds
        self._data.update({'e_bounds': e_bounds})
        return None

    def get_e_bounds(self):
        return self._e_bounds

    def set_spread_to_error_bound(self, s2e_ratio):
        self._s2e_ratio = s2e_ratio
        self._data.update({'s2e_ratio': s2e_ratio})
        return None

    def get_spread_to_error_bound(self):
        return self._s2e_ratio
