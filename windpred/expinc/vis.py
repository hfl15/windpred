import os
import numpy as np
import matplotlib.pyplot as plt

from windpred.utils.base import tag_path, make_dir, DIR_LOG
from windpred.utils.data_parser import DataGenerator
from windpred.utils.model_base import DefaultConfig, MONTH_LIST

from matplotlib.backends.backend_pdf import PdfPages


if __name__ == '__main__':
    tag = tag_path(os.path.abspath(__file__), 2)

    target_size = DefaultConfig.target_size
    period = DefaultConfig.period
    window = DefaultConfig.window
    train_step = DefaultConfig.train_step
    test_step = DefaultConfig.test_step
    single_step = DefaultConfig.single_step
    obs_data_path_list = DefaultConfig.obs_data_path_list

    target = 'V'
    i_run = 0
    month = str(MONTH_LIST[-1])

    dir_log = os.path.join(DIR_LOG, tag, target, str(i_run))
    make_dir(dir_log)

    data_generator_list = []
    for obs_data_path in obs_data_path_list:
        data_generator = DataGenerator(period, window, path=obs_data_path)
        data_generator.prepare_data(target_size, train_step=train_step, test_step=test_step, single_step=single_step)
        data_generator_list.append(data_generator)

    for data_generator in data_generator_list:
        station_name = data_generator.station_name

        _, nwp, obs, _ = data_generator.extract_evaluation_data(target)
        plt.plot(obs, label='TRUTH')
        plt.plot(nwp, label='NWP')

        mhstn_cnn_covar_path = 'expinc_mhstn_covar/{}/{}/{}/y_pred_{}_combine_module_conv.txt'.format(
            target, month, str(i_run), station_name)
        mhstn_cnn_covar_pred = np.loadtxt(os.path.join(DIR_LOG, mhstn_cnn_covar_path))
        plt.plot(mhstn_cnn_covar_pred, label='MHSTN')

        plt.legend(loc='best')
        plt.ylabel('Value (meter/second)')
        plt.xlabel('Time (hours)')
        plt.tight_layout()

        plt.savefig(os.path.join(dir_log, "{}".format(station_name)), dpi=750, bbox_inches='tight')

        pdf = PdfPages(os.path.join(dir_log, "{}.pdf".format(station_name)))
        pdf.savefig()
        pdf.close()

        plt.close()














