

class DEUS_ASSERT:
    @classmethod
    def has(cls, what, who, who_name="", use_also=False):
        assert isinstance(what, list), "wrong type"
        assert isinstance(who, dict), "wrong type"
        assert isinstance(who_name, str), "wrong type"

        if all(key in who.keys() for key in what):
            pass
        else:
            assert False, cls.__failure_msg_for_has(what, who_name, use_also)

    @classmethod
    def __failure_msg_for_has(cls, what, who_name, use_also):
        msg = "Wrong keys passed...\nThe "
        if who_name == "":
            pass
        else:
             msg += "\'" + who_name + "\' "

        if use_also:
            msg += "keys must also include:\n"
        else:
            msg += "keys must include:\n"

        for i, key in enumerate(what):
            if i + 1 < len(what):
                msg += "\'" + key + "\', "
            else:
                msg += "\'" + key + "\'.\n"

        msg += "Look for typos, white spaces, and missing keys."
        return msg
