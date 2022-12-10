import os

from windpred.utils.model_base import DefaultConfig, BaseLSTM, BaseMLP
from windpred.utils import exp_dir
from windpred.baseline import temporal_nn


def run(target, tag, feature_mode, eval_mode, model_cls):
    if target == 'DIR':
        tag_file_list = [None]
        file_exp_in = os.path.join(tag, feature_mode)
        for mode in ['run', 'reduce']:
            exp_dir.main(mode, DefaultConfig(), eval_mode, file_exp_in, tag_file_list)
    else:
        for mode in ['run', 'reduce']:
            temporal_nn.main(tag, DefaultConfig(), target, mode+'-'+feature_mode, eval_mode, model_cls)


def run_lstm(target, tag, feature_mode, eval_mode):
    run(target, tag, feature_mode, eval_mode, BaseLSTM)


def run_mlp(target, tag, feature_mode, eval_mode):
    run(target, tag, feature_mode, eval_mode, BaseMLP)



