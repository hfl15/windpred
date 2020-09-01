import os

from windpred.utils.base import DIR_LOG, make_dir
from windpred.utils.data_parser import DataGeneratorV2Spatial
from windpred.utils.model_base import batch_run
from windpred.utils.model_base import TESTING_SLIDING_WINDOW, MONTH_LIST, get_month_list, reduce


def main(target, mode, eval_mode, config, tag, func, csv_result_list=None):
    target_size = config.target_size
    period = config.period
    window = config.window
    train_step = config.train_step
    test_step = config.test_step
    single_step = config.single_step
    norm = config.norm
    x_divide_std = config.x_divide_std
    n_epochs = config.n_epochs
    n_runs = config.n_runs
    station_name_list = config.station_name_list

    dir_log_target = os.path.join(DIR_LOG, tag, target)
    make_dir(dir_log_target)

    if mode.startswith('run'):
        data_generator_spatial = DataGeneratorV2Spatial(period, window, norm=norm, x_divide_std=x_divide_std)
        for wid in range(TESTING_SLIDING_WINDOW, len(MONTH_LIST)):
            dir_log_exp = os.path.join(dir_log_target, str(MONTH_LIST[wid]))
            months = get_month_list(eval_mode, wid)
            data_generator_spatial.set_data(months)
            data_generator_spatial.prepare_data(target_size,
                                                train_step=train_step, test_step=test_step, single_step=single_step)
            batch_run(n_runs, dir_log_exp,
                      lambda dir_log_curr:
                      func(station_name_list, dir_log_curr, data_generator_spatial, target, n_epochs))
    elif mode.startswith('reduce') and csv_result_list is not None:
        reduce(csv_result_list, target, dir_log_target, n_runs, station_name_list)
    else:
        raise ValueError("The setting can not be found!")
