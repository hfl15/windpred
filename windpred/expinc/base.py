import os
from windpred.utils.base import tag_path

eval_mode = 'increment'


def get_tag(file):
    return tag_path(os.path.abspath(file), 2)

