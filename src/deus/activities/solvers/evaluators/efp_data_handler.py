from deus.activities.solvers.evaluators import EvaluationDataHandler


class EFPEvalDataHandler(EvaluationDataHandler):
    def __init__(self, eval_path, p_samples):
        super().__init__(eval_path)
        self._data.update({'n_model_evals': 0})

        self._p_samples = None
        self._set_p_samples(p_samples)

        self._worst_efp = None

    def _set_p_samples(self, p_samples):
        self._p_samples = p_samples
        self._data.update({'p_samples': p_samples})

    def get_p_samples(self):
        return self._p_samples

    def set_worst_efp(self, worst):
        self._worst_efp = worst
        self._data.update({'worst_efp': worst})

    def get_worst_efp(self):
        return self._worst_efp
