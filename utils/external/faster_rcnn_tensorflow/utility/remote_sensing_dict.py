# -*- coding: utf-8 -*-

NAME_LABEL_MAP = {
    'back_ground': 0,
    'building': 1
}


def get_label_name_map():
    reverse_dict = {}
    for name, label in NAME_LABEL_MAP.items():
        reverse_dict[label] = name
    return reverse_dict

LABEL_NAME_MAP = get_label_name_map()