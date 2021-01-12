import os

from windpred.utils.base import tag_path
from windpred.utils.model_base import DefaultConfig
from windpred.utils import exp_dir

from windpred.utils.exp import main_spatial
from windpred.baseline import lstm_spatial

from windpred.exp.base_lstm_spatial import run
from windpred.expinc.base import *


if __name__ == '__main__':
    tag = get_tag(__file__)
    target = 'V'
    run(target, tag, eval_mode)

    # if target == 'DIR':
    #     tag_file_list = [None]
    #     exp_dir.main_old('run', eval_mode, tag, tag_file_list)
    #     exp_dir.main_old('reduce', eval_mode, tag, tag_file_list)
    # else:
    #     features_history, features_future = [target], ['NEXT_NWP_{}'.format(target)]
    #     concat_mode = 'parallel'
    #     model_mode = 'lstm1'
    #     csv_result_list = ['metrics_model.csv', 'metrics_nwp.csv']
    #
    #     for mode in ['run', 'reduce']:
    #         func = lstm_spatial.run_lstm(features_history, features_future, concat_mode, model_mode)
    #         main_spatial(target, mode, eval_mode, DefaultConfig, tag, func, csv_result_list)
