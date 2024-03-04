from deus.activities.solvers.evaluators import EvaluationDataHandler


class DSScoreEvalDataHandler(EvaluationDataHandler):
    def __init__(self, eval_path, p_best):
        super().__init__(eval_path)
        self._p_best = None
        self.set_p_best(p_best)

    def set_p_best(self, p_best):
        self._p_best = p_best
        self._data.update({'p_best': p_best})

    def get_p_best(self):
        return self._p_best
