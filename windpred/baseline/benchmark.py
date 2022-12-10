"""
    persistence model and NWP model.
"""

import os
import numpy as np
import pandas as pd

from windpred.utils.base import make_dir
from windpred.utils.data_parser import DataGenerator
from windpred.utils.evaluation import Evaluator, EvaluatorDir
from windpred.utils.model_base import DefaultConfig, TESTING_SLIDING_WINDOW, MONTH_LIST, get_month_list, reduce_multiple_splits


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


def main(target, tag, config: DefaultConfig, dir_log, eval_mode):
    dir_log_target = os.path.join(dir_log, tag, target)
    make_dir(dir_log_target)

    csv_list = ['metrics_model.csv', 'metrics_nwp.csv']

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

        # copy the evaluation results
        for csv in csv_list:
            df = pd.read_csv(os.path.join(dir_log_exp, 'evaluate', csv), index_col=0)
            df.to_csv(os.path.join(dir_log_exp, csv))

    reduce_multiple_splits(dir_log_target, csv_list)






