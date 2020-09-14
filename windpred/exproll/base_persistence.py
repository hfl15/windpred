"""
    use the observations of current day to predict the next day.
"""

import os
import numpy as np
import pandas as pd

from windpred.utils.base import DIR_LOG
from windpred.utils.base import tag_path, make_dir
from windpred.utils.data_parser import DataGenerator
from windpred.utils.evaluation import Evaluator, EvaluatorDir
from windpred.utils.model_base import DefaultConfig
from windpred.utils.model_base import TESTING_SLIDING_WINDOW, MONTH_LIST, get_month_list

from windpred.baseline.persistence import main

from windpred.exproll.base import eval_mode


if __name__ == '__main__':
    tag = tag_path(os.path.abspath(__file__), 2)

    target = 'V'
    main(tag, DefaultConfig, DIR_LOG, eval_mode)

    # target_size = DefaultConfig.target_size
    # period = DefaultConfig.period
    # window = DefaultConfig.window
    # train_step = DefaultConfig.train_step
    # test_step = DefaultConfig.test_step
    # single_step = DefaultConfig.single_step
    # obs_data_path_list = DefaultConfig.obs_data_path_list
    #
    # target = 'V'
    # dir_log_target = os.path.join(DIR_LOG, tag, target)
    # make_dir(dir_log_target)
    #
    # data_generator_list = []
    # for obs_data_path in obs_data_path_list:
    #     data_generator = DataGenerator(period, window, path=obs_data_path)
    #     data_generator_list.append(data_generator)
    #
    # for wid in range(TESTING_SLIDING_WINDOW, len(MONTH_LIST)):
    #     dir_log_exp = os.path.join(dir_log_target, str(MONTH_LIST[wid]))
    #     months = get_month_list(eval_mode, wid)
    #     for data_generator in data_generator_list:
    #         data_generator.set_data(months)
    #         data_generator.prepare_data(target_size, train_step=train_step, test_step=test_step, single_step=single_step)
    #     run(data_generator_list, dir_log_exp, target)
    #
    # csv_list = ['metrics_model.csv', 'metrics_nwp.csv']
    # if target == 'DIR':
    #     reduce_multiple_splits_dir(dir_log_target, csv_list)
    # else:
    #     reduce_multiple_splits(dir_log_target, csv_list)









