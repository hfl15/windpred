import os
import numpy as np
import matplotlib.pyplot as plt

from windpred.utils.base import tag_path, make_dir, DIR_LOG
from windpred.utils.evaluation import cal_delta
from windpred.utils.data_parser import DataGenerator
from windpred.utils.model_base import DefaultConfig, MONTH_LIST

from matplotlib.backends.backend_pdf import PdfPages


def plot(tag, config, target, mhstn_root):
    i_run = 0
    month = str(MONTH_LIST[-1])
    dir_log = make_dir(os.path.join(DIR_LOG, tag, target, str(i_run)))

    data_generator_list = []
    for obs_data_path in config.obs_data_path_list:
        data_generator = DataGenerator(config.period, config.window, path=obs_data_path)
        data_generator.prepare_data(config.target_size, train_step=config.train_step, test_step=config.test_step,
                                    single_step=config.single_step)
        data_generator_list.append(data_generator)

    for data_generator in data_generator_list:
        station_name = data_generator.station_name

        _, nwp, obs, _ = data_generator.extract_evaluation_data(target)
        plt.plot(obs, label='TRUTH')
        plt.plot(nwp, label='NWP')

        mhstn_path = '{}/{}/{}/{}/y_pred_{}_combine_module_conv.txt'.format(
            mhstn_root, target, month, str(i_run), station_name)
        mhstn_pred = np.loadtxt(os.path.join(DIR_LOG, mhstn_path))
        plt.plot(mhstn_pred, label='MHSTN')

        plt.legend(loc='best')
        plt.ylabel('Value (meter/second)')
        plt.xlabel('Time (hours)')
        plt.tight_layout()

        plt.savefig(os.path.join(dir_log, "{}".format(station_name)), dpi=750, bbox_inches='tight')

        pdf = PdfPages(os.path.join(dir_log, "{}.pdf".format(station_name)))
        pdf.savefig()
        pdf.close()

        plt.close()


def plot_dir(tag, config, target, mhstn_root):
    i_run = 0
    month = str(MONTH_LIST[-1])

    dir_log = make_dir(os.path.join(DIR_LOG, tag, target, str(i_run)))

    data_generator_list = []
    for obs_data_path in config.obs_data_path_list:
        data_generator = DataGenerator(config.period, config.window, path=obs_data_path)
        data_generator.prepare_data(config.target_size, train_step=config.train_step, test_step=config.test_step,
                                    single_step=config.single_step)
        data_generator_list.append(data_generator)

    for data_generator in data_generator_list:
        station_name = data_generator.station_name

        _, nwp, obs, _ = data_generator.extract_evaluation_data(target)
        mhstn_path = '{}/{}/{}/{}/y_pred_{}_combine_module_conv.txt'.format(
            mhstn_root, target, month, str(i_run), station_name)
        mhstn_pred = np.loadtxt(os.path.join(DIR_LOG, mhstn_path))

        delta_nwp = cal_delta(obs, nwp)
        delta_cnn = cal_delta(obs, mhstn_pred)

        grid = plt.GridSpec(3, 1)

        ax_main = plt.subplot(grid[0:2, 0])
        plt.plot(obs, label='TRUTH')
        plt.plot(nwp, label='NWP')
        plt.plot(mhstn_pred, label='MH-STDNN')
        plt.legend(loc='best')
        plt.ylabel('Value (degree)')

        ax_err = plt.subplot(grid[2, 0], sharex=ax_main)
        plt.plot(np.zeros(len(delta_nwp)))
        plt.plot(delta_nwp)
        plt.plot(delta_cnn)
        plt.ylabel('Error (degree)')
        plt.xlabel('Time (hours)')

        plt.savefig(os.path.join(dir_log, "{}".format(station_name)), dpi=750, bbox_inches='tight')

        pdf = PdfPages(os.path.join(dir_log, "{}.pdf".format(station_name)))
        pdf.savefig()
        pdf.close()

        plt.close()


if __name__ == '__main__':
    tag = tag_path(os.path.abspath(__file__), 2)

    plot(tag, DefaultConfig(), 'V', 'expinc_mhstn_covar')
    plot(tag, DefaultConfig(), 'VX', 'expinc_mhstn')
    plot(tag, DefaultConfig(), 'VY', 'expinc_mhstn')
    plot_dir(tag, DefaultConfig(), 'DIR', 'expinc_mhstn_covar')


    # data_generator_list = []
    # for obs_data_path in config.obs_data_path_list:
    #     data_generator = DataGenerator(config.period, config.window, path=obs_data_path)
    #     data_generator.prepare_data(config.target_size, train_step=config.train_step, test_step=config.test_step,
    #                                 single_step=config.single_step)
    #     data_generator_list.append(data_generator)
    #
    # for data_generator in data_generator_list:
    #     station_name = data_generator.station_name
    #
    #     _, nwp, obs, _ = data_generator.extract_evaluation_data(target)
    #     plt.plot(obs, label='TRUTH')
    #     plt.plot(nwp, label='NWP')
    #
    #     mhstn_cnn_covar_path = 'expinc_mhstn_covar/{}/{}/{}/y_pred_{}_combine_module_conv.txt'.format(
    #         target, month, str(i_run), station_name)
    #     mhstn_cnn_covar_pred = np.loadtxt(os.path.join(DIR_LOG, mhstn_cnn_covar_path))
    #     plt.plot(mhstn_cnn_covar_pred, label='MHSTN')
    #
    #     plt.legend(loc='best')
    #     plt.ylabel('Value (meter/second)')
    #     plt.xlabel('Time (hours)')
    #     plt.tight_layout()
    #
    #     plt.savefig(os.path.join(dir_log, "{}".format(station_name)), dpi=750, bbox_inches='tight')
    #
    #     pdf = PdfPages(os.path.join(dir_log, "{}.pdf".format(station_name)))
    #     pdf.savefig()
    #     pdf.close()
    #
    #     plt.close()














