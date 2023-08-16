def list_of_dicts_to_dict_of_lists(list_of_dicts):
    dict_of_lists = {}
    for dictionary in list_of_dicts:
        for key, value in dictionary.items():
            if key not in dict_of_lists:
                dict_of_lists[key] = []
            dict_of_lists[key].append(value)
    return dict_of_lists
