"""
    persistence model and NWP model.
"""

import os
import numpy as np
import pandas as pd

from windpred.utils.base import make_dir
from windpred.utils.data_parser import DataGenerator
from windpred.utils.evaluation import Evaluator, EvaluatorDir
from windpred.utils.model_base import DefaultConfig
from windpred.utils.model_base import TESTING_SLIDING_WINDOW, MONTH_LIST, get_month_list


def run(data_generator_list, dir_log, target):
    if target == 'DIR':
        evaluator_model = EvaluatorDir(dir_log, 'model')
        evaluator_nwp = EvaluatorDir(dir_log, 'nwp')
    else:
        evaluator_model = Evaluator(dir_log, 'model')
        evaluator_nwp = Evaluator(dir_log, 'nwp')
    for data_generator in data_generator_list:
        station_name = data_generator.station_name
        (_, _), (_, _), (x_test, y_test) = \
            data_generator.extract_training_data(x_attributes=[target], y_attributes=[target])
        y_pred = x_test.squeeze().ravel()

        speed, nwp, obs, filter_big_wind = data_generator.extract_evaluation_data(target)
        evaluator_model.append(obs, y_pred, filter_big_wind, key=station_name)
        evaluator_nwp.append(obs, nwp, filter_big_wind, key=station_name)

        np.savetxt(os.path.join(dir_log, '{}.txt'.format(station_name)), y_pred)


def reduce_multiple_splits(path, csvf_list, col_name='all_rmse'):
    for csvf in csvf_list:
        dir_list = sorted([d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))])
        df_list = []
        for month in dir_list:
            df = pd.read_csv(os.path.join(path, month, 'evaluate', csvf), index_col=0)
            df = df[col_name]
            df_list.append(df)
        dfc = pd.concat(df_list, axis=1, keys=dir_list)
        dfc.to_csv(os.path.join(path, csvf))
        print("Finish to processing {}".format(csvf))


def reduce_multiple_splits_dir(path, csvf_list, col_name='all_mae'):
    reduce_multiple_splits(path, csvf_list, col_name)


def main(tag, config: DefaultConfig, dir_log, eval_mode):
    # target_size = config.target_size
    # period = config.period
    # window = config.window
    # train_step = config.train_step
    # test_step = config.test_step
    # single_step = config.single_step
    # obs_data_path_list = config.obs_data_path_list

    target = 'V'
    dir_log_target = os.path.join(dir_log, tag, target)
    make_dir(dir_log_target)

    data_generator_list = []
    for obs_data_path in config.obs_data_path_list:
        data_generator = DataGenerator(config.period, config.window, path=obs_data_path)
        data_generator_list.append(data_generator)

    for wid in range(TESTING_SLIDING_WINDOW, len(MONTH_LIST)):
        dir_log_exp = os.path.join(dir_log_target, str(MONTH_LIST[wid]))
        months = get_month_list(eval_mode, wid)
        for data_generator in data_generator_list:
            data_generator.set_data(months)
            data_generator.prepare_data(config.target_size, train_step=config.train_step, test_step=config.test_step,
                                        single_step=config.single_step)
        run(data_generator_list, dir_log_exp, target)

    csv_list = ['metrics_model.csv', 'metrics_nwp.csv']
    if target == 'DIR':
        reduce_multiple_splits_dir(dir_log_target, csv_list)
    else:
        reduce_multiple_splits(dir_log_target, csv_list)





