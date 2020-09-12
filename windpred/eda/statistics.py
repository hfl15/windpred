import os

from windpred.utils.base import tag_path, make_dir, DIR_LOG
from windpred.utils.data_parser import DataGenerator
from windpred.utils.model_base import DefaultConfig


def describe(data_generator_list, dir_log):
    df_des_sum = None
    for i, generator in enumerate(data_generator_list):
        station_name = generator.station_name
        df = generator.df_origin.copy()
        # df = df[generator.train_start_idx:generator.train_end_idx]
        df_des = df.describe()
        df_des.to_csv(os.path.join(dir_log, '{}.csv'.format(station_name)))

        if i == 0:
            df_des_sum = df_des
        else:
            df_des_sum += df_des

    df_des_mean = df_des_sum / len(data_generator_list)
    df_des_mean.to_csv(os.path.join(dir_log, 'mean.csv'))


def main(tag):
    target_size = DefaultConfig.target_size
    period = DefaultConfig.period
    window = DefaultConfig.window
    train_step = DefaultConfig.train_step
    test_step = DefaultConfig.test_step
    single_step = DefaultConfig.single_step
    obs_data_path_list = DefaultConfig.obs_data_path_list

    dir_log = os.path.join(DIR_LOG, tag)
    make_dir(dir_log)

    data_generator_list = []
    for obs_data_path in obs_data_path_list:
        data_generator = DataGenerator(period, window, path=obs_data_path)
        data_generator.prepare_data(target_size, train_step=train_step, test_step=test_step, single_step=single_step)
        data_generator_list.append(data_generator)

    describe(data_generator_list, dir_log)


if __name__ == '__main__':
    tag = tag_path(os.path.abspath(__file__), 2)
    main(tag)










