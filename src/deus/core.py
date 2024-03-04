import glob
import pickle
from os import makedirs, remove
from os.path import exists

from deus.utils.assertions import DEUS_ASSERT

from deus.activities import \
    DesignSpaceManager, \
    ParameterEstimationManager, \
    SetMembershipEstimationManager


class DEUS:
    def __init__(self, activity_form):
        if self.check_is_ok_for(activity_form):
            self.activity_form = activity_form
            self.activity_manager = None

            self.cs_path = self.activity_form["activity_settings"]["case_path"]
            self.cs_name = self.activity_form["activity_settings"]["case_name"]
            self.cs_folder = self.cs_path + "/" + self.cs_name + "/"

            to_resume = self.activity_form["activity_settings"]["resume"]
            if exists(self.cs_folder):
                if not to_resume:
                    self.delete_all_files_in_cs_folder()
            else:
                try:
                    makedirs(self.cs_folder)
                except OSError:
                    print("Error: Creating the case study directory:\n",
                          "cs_folder")

            with open(self.cs_folder + 'activity_form.pkl', 'wb') as file:
                pickle.dump(activity_form, file)

            # Are we resuming the activity mentioned within the form?
            if to_resume:
                with open(self.cs_folder + 'solution_state.pkl', 'rb') as file:
                    self.activity_manager = pickle.load(file)
            else:
                activity = self.activity_form["activity_type"]
                if activity == "pe":
                    self.activity_manager = ParameterEstimationManager(
                        self.activity_form)
                elif activity == "dsc":
                    self.activity_manager = DesignSpaceManager(
                        self.activity_form)
                elif activity == "sme":
                    self.activity_manager = SetMembershipEstimationManager(
                        self.activity_form)
                else:
                    assert False, "Unrecognized activity."

    def solve(self):
        self.activity_manager.solve_problem()

    @staticmethod
    def check_is_ok_for(a_form):
        assert isinstance(a_form, dict), \
            "The activity form must be a dictionary."

        mkeys = ['activity_type', 'activity_settings', 'problem', 'solver']
        DEUS_ASSERT.has(mkeys, a_form, "activity form")
        return True

    def delete_all_files_in_cs_folder(self):
        files = glob.glob(self.cs_folder + '*')
        for file in files:
            remove(file)
