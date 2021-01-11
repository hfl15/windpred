import os

from windpred.utils.base import tag_path
from windpred.utils.model_base import DefaultConfig

from windpred.utils import exp_dir
from windpred.mhstn.base import get_covariates_history
from windpred.baseline import convlstm

from windpred.expinc.base import *
from windpred.exp.base_convlstm import run_covar

if __name__ == '__main__':
    tag = get_tag(__file__)
    target = 'V'
    run_covar(target, tag, eval_mode)

    # target = 'V'
    # model_name = 'convlstm'
    #
    # if target == 'DIR':
    #     tag_file_list = [model_name]
    #     exp_dir.main_old('run', eval_mode, tag, tag_file_list)
    #     exp_dir.main_old('reduce', eval_mode, tag, tag_file_list)
    # else:
    #     features_history = get_covariates_history(target)
    #     features_future = ['NEXT_NWP_{}'.format(target)]
    #     for mode in ['run', 'reduce']:
    #         convlstm.main(target, mode, eval_mode, DefaultConfig, tag, model_name, features_history, features_future)
