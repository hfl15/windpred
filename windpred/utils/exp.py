import os
import shutil

from .base import DIR_LOG, make_dir
from .data_parser import DataGeneratorSpatial
from .model_base import TESTING_SLIDING_WINDOW, MONTH_LIST, get_month_list, batch_run
from .model_base import reduce, DefaultConfig


def get_covariates_history_all():
    return ['V', 'VX', 'VY', 'DIRRadian', 'SLP', 'TP', 'RH']


def get_covariates_future_all():
    return ['NEXT_NWP_{}'.format(feat) for feat in ['V', 'VX', 'VY', 'DIRRadian', 'SLP', 'TP', 'RH']]


def main_spatial(target, mode, eval_mode, config:DefaultConfig, tag, func, csv_result_list=None):
    dir_log_target = os.path.join(DIR_LOG, tag, target)
    make_dir(dir_log_target)

    if mode.startswith('run'):
        data_generator_spatial = DataGeneratorSpatial(config.period, config.window, norm=config.norm,
                                                      x_divide_std=config.x_divide_std)
        for wid in range(TESTING_SLIDING_WINDOW, len(MONTH_LIST)):
            dir_log_exp = os.path.join(dir_log_target, str(MONTH_LIST[wid]))
            months = get_month_list(eval_mode, wid)
            data_generator_spatial.set_data(months)
            data_generator_spatial.prepare_data(config.target_size,
                                                train_step=config.train_step, test_step=config.test_step,
                                                single_step=config.single_step)
            batch_run(config.n_runs, dir_log_exp,
                      lambda dir_log_curr:
                      func(config.station_name_list, dir_log_curr, data_generator_spatial, target, config.n_epochs))
    elif mode.startswith('reduce') and csv_result_list is not None:
        reduce(csv_result_list, target, dir_log_target, config.n_runs, config.station_name_list)
    elif mode.startswith('clear'):
        paths = [p for p in os.listdir(dir_log_target) if not p.endswith('.csv')]
        for p in paths:
            shutil.rmtree(os.path.join(dir_log_target, p))
    else:
        raise ValueError("The setting can not be found!")


def main_spatial_duq(target, mode, eval_mode, config: DefaultConfig, tag, func, csv_result_list=None):
    dir_log_target = make_dir(os.path.join(DIR_LOG, tag, target))

    if mode.startswith('run'):
        data_generator_spatial = DataGeneratorSpatial(config.period, config.window, norm=config.norm,
                                                      x_divide_std=config.x_divide_std)
        for wid in range(TESTING_SLIDING_WINDOW, len(MONTH_LIST)):
            dir_log_exp = make_dir(os.path.join(dir_log_target, str(MONTH_LIST[wid])))
            months = get_month_list(eval_mode, wid)
            data_generator_spatial.set_data(months)
            data_generator_spatial.prepare_data(config.target_size,
                                                train_step=config.train_step, test_step=config.test_step,
                                                single_step=config.single_step)
            batch_run(config.n_runs, dir_log_exp,
                      lambda dir_log_curr:
                      func(config, dir_log_curr, data_generator_spatial, target))
    elif mode.startswith('reduce') and csv_result_list is not None:
        reduce(csv_result_list, target, dir_log_target, config.n_runs, config.station_name_list)
    elif mode.startswith('clear'):
        paths = [p for p in os.listdir(dir_log_target) if not p.endswith('.csv')]
        for p in paths:
            shutil.rmtree(os.path.join(dir_log_target, p))
    else:
        raise ValueError("The setting can not be found!")
