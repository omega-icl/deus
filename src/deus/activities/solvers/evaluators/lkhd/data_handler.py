from deus.activities.solvers.evaluators import EvaluationDataHandler


class LogLkhdEvalDataHandler(EvaluationDataHandler):
    def __init__(self, eval_path):
        super().__init__(eval_path)
