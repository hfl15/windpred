import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

from windpred.utils.base import tag_path, make_dir, DIR_LOG
from windpred.utils.data_parser import DataGenerator
from windpred.utils.model_base import DefaultConfig
from windpred.utils.exp import get_covariates_history_all


def visualize(data_generator_list, dir_log):
    for i, generator in enumerate(data_generator_list):
        station_name = generator.station_name
        df = generator.df_origin.copy()
        df = df[:24*30]
        for col in df.columns:
            if df[col].dtype == np.float or df[col].dtype == np.int:
                plt.plot(df[col])
                plt.xlabel('Time (hours)')
                plt.tight_layout()
                pdf = PdfPages(os.path.join(dir_log, '{}_{}.pdf'.format(station_name, col)))
                pdf.savefig()
                pdf.close()
                plt.clf()


def visualize_couple(data_generator_list, dir_log):
    for i, generator in enumerate(data_generator_list):
        station_name = generator.station_name
        df = generator.df_origin.copy()
        df = df[:24*30]
        vars_history = get_covariates_history_all() + ['DIR']
        vars_future = ['NWP_{}'.format(v) for v in vars_history]
        for v_h, v_f in zip(vars_history, vars_future):
            plt.plot(df[v_h], label='TRUTH')
            plt.plot(df[v_f], label='NWP')
            plt.legend(loc='best')
            plt.xlabel('Time (hours)')
            plt.tight_layout()
            pdf = PdfPages(os.path.join(dir_log, '{}_{}.pdf'.format(station_name, v_h)))
            pdf.savefig()
            pdf.close()
            plt.clf()


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

    # visualize(data_generator_list, dir_log)
    visualize_couple(data_generator_list, dir_log)


if __name__ == '__main__':
    tag = tag_path(os.path.abspath(__file__), 2)
    main(tag)










