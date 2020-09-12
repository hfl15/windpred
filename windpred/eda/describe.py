import sys
sys.path.append('..')

import os
import pandas as pd
import numpy as np

from utilities.utils import tag_path, make_dir, get_station_name, get_missing_tag
from utilities.prepare_data import DataGeneratorV2, get_files_path

from paper.base import DIR_LOG, DefaultConfig


def load_data(path):
    df = pd.read_csv(path)
    df['DateTime'] = pd.to_datetime(df['DateTime'])
    columns = list(df.columns)
    columns.remove('DateTime')
    df[columns] = df[columns].astype(np.float)
    print("Finish to load data from path={}:".format(path), df.shape)
    return df


if __name__ == '__main__':
    tag = tag_path(os.path.abspath(__file__), 2)

    target_size = DefaultConfig.target_size
    period = DefaultConfig.period
    window = DefaultConfig.window
    train_step = DefaultConfig.train_step
    test_step = DefaultConfig.test_step
    single_step = DefaultConfig.single_step
    n_epochs = DefaultConfig.n_epochs
    n_runs = DefaultConfig.n_runs
    obs_data_path_list = DefaultConfig.obs_data_path_list
    nwp_path = DefaultConfig.nwp_path

    dir_log = os.path.join(DIR_LOG, tag)

    df_obs_list = {}
    for path in obs_data_path_list:
        station_name = get_station_name(path)
        df = load_data(path)
        df_obs_list[station_name] = df
    df_nwp = load_data(nwp_path)

    # missing value
    print("**** check data")
    print("** station")
    TAG_MISSING = get_missing_tag()

    def check_spd(df_spd):
        spd_outliers = np.where((df_spd.values < 0) & (df_spd.values > 15))[0]
        print("spd outliers: ", spd_outliers, df_spd[spd_outliers])

    def check_missing(df):
        res_miss = {}
        res_miss_ratio = {}
        for col in df:
            df_col = df[col]
            miss_inds = np.where(df_col.values == TAG_MISSING)[0]
            res_miss[col] = len(miss_inds)
            res_miss_ratio[col] = res_miss[col] / df.shape[0]
        print("missing value:")
        print(res_miss)
        print(res_miss_ratio)

    for key, df in df_obs_list.items():
        print("** station={}".format(key))
        check_spd(df['SPD10'])
        check_missing(df)
    print()
    print("** NWP")
    check_spd(df_nwp['SPD10'])
    check_missing(df_nwp)

    print()
    print()
    data_generator_list = []
    for obs_data_path in obs_data_path_list:
        data_generator = DataGeneratorV2(period, window, path=obs_data_path)
        data_generator.prepare_data(target_size, train_step=train_step, test_step=test_step, single_step=single_step)
        data_generator_list.append(data_generator)

