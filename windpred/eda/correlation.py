import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics import tsaplots

from windpred.utils.base import tag_path, make_dir, DIR_LOG
from windpred.utils.data_parser import DataGenerator
from windpred.utils.model_base import DefaultConfig

TARGET_VARIABLES = ['V', 'DIR', 'VX', 'VY']


def corr_with_nwp(data_generator_list, dir_log):
    station_name_list = []
    for generator in data_generator_list:
        station_name = generator.station_name
        df = generator.df_origin.copy()
        df = df[generator.train_start_idx:generator.train_end_idx]
        df.corr().to_csv(os.path.join(dir_log, '{}.csv'.format(station_name)))
        station_name_list.append(station_name)

    def get_corr_target(target):
        corr = []
        for station_name in station_name_list:
            df_corr = pd.read_csv(os.path.join(dir_log, '{}.csv'.format(station_name)), index_col=0)
            corr.append(df_corr.loc['{}'.format(target), 'NWP_{}'.format(target)])
        return corr

    res = {}
    for target in TARGET_VARIABLES:
        corr = np.round(get_corr_target(target), 4)
        res[target] = corr
    df_corr = pd.DataFrame(res, index=station_name_list)
    df_corr.to_csv(os.path.join(dir_log, 'reduce.csv'))


def corr_variables(data_generator_list, dir_log):
    station_name_list = []
    for generator in data_generator_list:
        station_name = generator.station_name
        df = generator.df_origin.copy()
        df = df[generator.train_start_idx:generator.train_end_idx]
        df.corr().to_csv(os.path.join(dir_log, '{}.csv'.format(station_name)))
        station_name_list.append(station_name)

    def get_corr_target(target):
        res = {}
        for station_name in station_name_list:
            df_corr = pd.read_csv(os.path.join(dir_log, '{}.csv'.format(station_name)), index_col=0)
            res[station_name] = np.round(df_corr.loc[:, '{}'.format(target)].values, 4)
        res = pd.DataFrame(res, index=list(df_corr.index))
        return res

    for target in TARGET_VARIABLES:
        get_corr_target(target).to_csv(os.path.join(dir_log, 'reduce_{}.csv'.format(target)))


def corr_spatial(data_generator_list, dir_log):
    def get_data(target):
        res = {}
        for generator in data_generator_list:
            station_name = generator.station_name
            df = generator.df_origin.copy()
            df = df[generator.train_start_idx:generator.train_end_idx]
            res[station_name] = df.loc[:, '{}'.format(target)].values
        res = pd.DataFrame(res)
        return res

    for target in TARGET_VARIABLES:
        df = get_data(target)
        df.corr().to_csv(os.path.join(dir_log, '{}.csv'.format(target)))

    for target in TARGET_VARIABLES:
        df = get_data(target)
        vals = df.values
        s1 = vals[:, 0]
        s2 = vals[:, 1:]
        res = {}
        for lag in range(25):
            s1_lag = s1[lag:]
            s2_lag = s2[:s1_lag.shape[0]]
            concat = np.hstack([s1_lag.reshape(s1_lag.shape[0], 1), s2_lag])
            corr = np.corrcoef(concat.transpose())
            res[lag] = corr[0]
        res = pd.DataFrame(res, index=list(df.columns))
        res.to_csv(os.path.join(dir_log, '{}_lag.csv'.format(target)))


def corr_auto(data_generator_list, dir_log):
    plt.rcParams.update({"font.size": 18, "font.family": 'SimHei'})  # 'font.weight': 'bold'
    for target in TARGET_VARIABLES:
        for generator in data_generator_list:
            station_name = generator.station_name
            df = generator.df_origin.copy()
            df = df[generator.train_start_idx:generator.train_end_idx]
            series = df[target].values
            fig = tsaplots.plot_acf(series, lags=48, title='')
            plt.savefig(os.path.join(dir_log, '{}_{}.png'.format(station_name, target)))
            plt.close(fig)


def main(tag, mode):
    target_size = DefaultConfig.target_size
    period = DefaultConfig.period
    window = DefaultConfig.window
    train_step = DefaultConfig.train_step
    test_step = DefaultConfig.test_step
    single_step = DefaultConfig.single_step
    obs_data_path_list = DefaultConfig.obs_data_path_list

    dir_log = os.path.join(DIR_LOG, tag, mode)
    make_dir(dir_log)

    data_generator_list = []
    for obs_data_path in obs_data_path_list:
        data_generator = DataGenerator(period, window, path=obs_data_path)
        data_generator.prepare_data(target_size, train_step=train_step, test_step=test_step, single_step=single_step)
        data_generator_list.append(data_generator)

    if mode == 'corr_with_nwp':
        corr_with_nwp(data_generator_list, dir_log)
    elif mode == 'corr_variables':
        corr_variables(data_generator_list, dir_log)
    elif mode == 'corr_spatial':
        corr_spatial(data_generator_list, dir_log)
    elif mode == 'corr_auto':
        corr_auto(data_generator_list, dir_log)
    else:
        raise ValueError('The mode = {} can not be found!'.format(mode))


if __name__ == '__main__':
    tag = tag_path(os.path.abspath(__file__), 2)
    for mode in ['corr_with_nwp', 'corr_variables', 'corr_spatial', 'corr_auto']:
        main(tag, mode)










