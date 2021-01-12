import os

from windpred.utils.base import tag_path

from windpred.utils.model_base import DefaultConfig, BaseMLP
from windpred.utils import exp_dir
from windpred.baseline import temporal_nn

from windpred.exp.base_nn import run_mlp
from windpred.expinc.base import *

if __name__ == '__main__':
    tag = tag_path(os.path.abspath(__file__), 2)
    target = 'V'
    run_mlp(target, tag, eval_mode)
    # feature_mode_list = ['history', 'future', 'history_future']
    # 
    # if target == 'DIR':
    #     for feature_mode in feature_mode_list:
    #         tag_file_list = [None]
    #         file_exp_in = os.path.join(tag, feature_mode)
    #         exp_dir.main_old('run', eval_mode, file_exp_in, tag_file_list)
    #         exp_dir.main_old('reduce', eval_mode, file_exp_in, tag_file_list)
    # else:
    #     for mode in ['run', 'reduce']:
    #         for feature_mode in feature_mode_list:
    #             temporal_nn.main(tag, DefaultConfig(), target, mode+'-'+feature_mode, eval_mode, BaseMLP)







