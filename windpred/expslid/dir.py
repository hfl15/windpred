import os
import numpy as np

from windpred.utils.base import DIR_LOG, make_dir, uv_to_degree_vec, plot_and_save_comparison
from windpred.utils.data_parser import DataGeneratorV2
from windpred.utils.evaluation import EvaluatorDir
from windpred.utils.model_base import DefaultConfig
from windpred.utils.model_base import TESTING_SLIDING_WINDOW, MONTH_LIST, get_month_list, reduce

# from windpred.expslid.base import eval_mode


def run_dir_from_uv(data_generator_list, dir_u, dir_v, dir_log, tag_file=None, n_runs=10, target='DIR10'):
    file_suffix = "" if tag_file is None else '_' + tag_file
    for i_run in range(n_runs):
        dir_log_curr = os.path.join(dir_log, str(i_run))
        make_dir(dir_log_curr)
        evaluator_model = EvaluatorDir(dir_log, 'model' + file_suffix)
        evaluator_nwp = EvaluatorDir(dir_log, 'nwp' + file_suffix)
        for data_generator in data_generator_list:
            station_name = data_generator.station_name
            speed, nwp, obs, filter_big_wind = data_generator.extract_evaluation_data(target)
            y_pred_u = np.loadtxt(os.path.join(dir_u, str(i_run), 'y_pred_{}.txt'.format(station_name+file_suffix)))
            y_pred_v = np.loadtxt(os.path.join(dir_v, str(i_run), 'y_pred_{}.txt'.format(station_name+file_suffix)))
            y_pred_dir = uv_to_degree_vec(y_pred_u, y_pred_v)

            plot_and_save_comparison(obs, y_pred_dir, dir_log_curr,
                                     filename='compare_{}.png'.format(station_name+file_suffix))
            evaluator_model.append(obs, y_pred_dir, filter_big_wind, key=station_name)
            evaluator_nwp.append(obs, nwp, filter_big_wind, key=station_name)
            np.savetxt(os.path.join(dir_log_curr, 'y_pred_{}.txt'.format(station_name+file_suffix)), y_pred_dir)


def main(mode, eval_mode, file_exp_in, tag_file_list):
    target_size = DefaultConfig.target_size
    period = DefaultConfig.period
    window = DefaultConfig.window
    train_step = DefaultConfig.train_step
    test_step = DefaultConfig.test_step
    single_step = DefaultConfig.single_step
    # n_epochs = DefaultConfig.n_epochs
    n_runs = DefaultConfig.n_runs
    obs_data_path_list = DefaultConfig.obs_data_path_list
    station_name_list = DefaultConfig.station_name_list

    target = 'DIR10'
    # mode = 'reduce'
    # file_exp_in = 'expslid_mhstn'
    dir_in = os.path.join(DIR_LOG, file_exp_in)
    dir_log_target = os.path.join(dir_in, target)
    make_dir(dir_log_target)

    # tag_file_list = [None]

    if mode.startswith('run'):
        data_generator_list = []
        for obs_data_path in obs_data_path_list:
            data_generator = DataGeneratorV2(period, window, path=obs_data_path)
            data_generator_list.append(data_generator)

        for wid in range(TESTING_SLIDING_WINDOW, len(MONTH_LIST)):
            dir_log_exp = os.path.join(dir_log_target, str(MONTH_LIST[wid]))
            months = get_month_list(eval_mode, wid)
            for data_generator in data_generator_list:
                data_generator.set_data(months)
                data_generator.prepare_data(target_size,
                                            train_step=train_step, test_step=test_step, single_step=single_step)
            dir_u = os.path.join(dir_in, 'U10', str(MONTH_LIST[wid]))
            dir_v = os.path.join(dir_in, 'V10', str(MONTH_LIST[wid]))

            for tag_file in tag_file_list:
                run_dir_from_uv(data_generator_list, dir_u, dir_v, dir_log_exp, tag_file, n_runs=n_runs)

    elif mode.startswith('reduce'):
        csv_result_list = []
        for tag_file in tag_file_list:
            csv = 'metrics_model.csv' if tag_file is None else 'metrics_model_{}.csv'.format(tag_file)
            csv_result_list.append(csv)
        reduce(csv_result_list, target, dir_log_target, n_runs, station_name_list)












