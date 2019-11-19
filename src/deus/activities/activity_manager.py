import abc

from deus.activities.interfaces import Observer


class ActivityManager(Observer):
    def __init__(self):
        pass

    @abc.abstractmethod
    def check_activity_type(self, a_type):
        raise NotImplementedError

    def check_settings(self, settings):
        assert isinstance(settings, dict), \
            "'activity_settings' must be a dictionary."

        mandatory_keys = ['case_name', 'case_path', 'resume', 'save_period']
        assert all(mkey in settings.keys() for mkey in mandatory_keys), \
            "The 'activity_settings' keys must be the following:\n" \
            "['case_name', 'case_path', 'resume', 'save_period']." \
            "Look for typos, white spaces or missing keys."

        assert settings["save_period"] > 0, \
            "'save_period' must be > 0."

    @abc.abstractmethod
    def check_problem(self, problem):
        raise NotImplementedError

    @abc.abstractmethod
    def solve_problem(self):
        raise NotImplementedError