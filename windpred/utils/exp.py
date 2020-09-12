import os

from .base import DIR_LOG, make_dir
from .data_parser import DataGeneratorSpatial
from .model_base import TESTING_SLIDING_WINDOW, MONTH_LIST, get_month_list, batch_run
from .model_base import reduce


def get_covariates_history(target):
    if target == 'V':
        features_history = ['V', 'VX', 'VY', 'SLP', 'TP']
    elif target == 'VX':
        features_history = ['V', 'VX', 'VY']
    elif target == 'VY':
        features_history = ['V', 'VX', 'VY']
    else:
        raise ValueError('The target={} can not be found!'.format(target))
    return features_history


def get_covariates_history_all():
    return ['V', 'VX', 'VY', 'DIRRadian', 'SLP', 'TP', 'RH']


def get_covariates_future_all():
    return ['NEXT_NWP_{}'.format(feat) for feat in ['V', 'VX', 'VY', 'DIRRadian', 'SLP', 'TP', 'RH']]


def main_spatial(target, mode, eval_mode, config, tag, func, csv_result_list=None):
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
        data_generator_spatial = DataGeneratorSpatial(period, window, norm=norm, x_divide_std=x_divide_std)
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