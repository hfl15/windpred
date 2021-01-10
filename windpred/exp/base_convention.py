import os

from windpred.utils.model_base import DefaultConfig
from windpred.utils import exp_dir

from windpred.baseline.convention import main, MODELS


def run(target, tag, eval_mode):
    if target == 'DIR':
        for model_name in MODELS.keys():
            file_in = os.path.join(tag, model_name)
            tag_file_list = [None]
            exp_dir.main('run', eval_mode, file_in, tag_file_list)
            exp_dir.main('reduce', eval_mode, file_in, tag_file_list)
    else:
        for model_name in MODELS.keys():
            main(tag, DefaultConfig(), target, 'run', eval_mode, model_name)
            main(tag, DefaultConfig(), target, 'reduce', eval_mode, model_name)

