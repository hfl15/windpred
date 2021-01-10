import os

from windpred.utils.model_base import DefaultConfig
from windpred.utils import exp_dir

from windpred.baseline import convention


def run(target, tag, eval_mode):
    if target == 'DIR':
        for model_name in convention.MODELS.keys():
            file_in = os.path.join(tag, model_name)
            tag_file_list = [None]
            for mode in ['run', 'reduce']:
                exp_dir.main(mode, DefaultConfig(), eval_mode, file_in, tag_file_list)
    else:
        for model_name in convention.MODELS.keys():
            for mode in ['reduce']:
            # for mode in ['run', 'reduce']:
                convention.main(tag, DefaultConfig(), target, mode, eval_mode, model_name)

