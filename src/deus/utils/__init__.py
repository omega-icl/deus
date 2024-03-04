from deus.utils.assertions import DEUS_ASSERT


def item_by_key(key_name, list_of_dicts):
    return [item for item in list_of_dicts if key_name in item.keys()][0]


def value_of(key_name, list_of_dicts):
    return [item for item in list_of_dicts if key_name in item.keys()][0][key_name]


def keys_in(list_of_dicts):
    keys = []
    for item in list_of_dicts:
        for k, v in item.items():
            keys.append(k)
    return keys
