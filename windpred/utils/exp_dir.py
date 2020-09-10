import os
import numpy as np

from windpred.utils.base import DIR_LOG, make_dir, vxy_to_dir_vec, plot_and_save_comparison
from windpred.utils.data_parser import DataGenerator
from windpred.utils.evaluation import EvaluatorDir
from windpred.utils.model_base import DefaultConfig
from windpred.utils.model_base import TESTING_SLIDING_WINDOW, MONTH_LIST, get_month_list, reduce


def run_dir_from_vxy(data_generator_list, dir_vx, dir_vy, dir_log, tag_file=None, n_runs=10, target='DIR'):
    file_suffix = "" if tag_file is None else '_' + tag_file
    for i_run in range(n_runs):
        dir_log_curr = os.path.join(dir_log, str(i_run))
        make_dir(dir_log_curr)
        evaluator_model = EvaluatorDir(dir_log_curr, 'model' + file_suffix)
        evaluator_nwp = EvaluatorDir(dir_log_curr, 'nwp' + file_suffix)
        for data_generator in data_generator_list:
            station_name = data_generator.station_name
            speed, nwp, obs, filter_big_wind = data_generator.extract_evaluation_data(target)
            y_pred_vx = np.loadtxt(os.path.join(dir_vx, str(i_run), 'y_pred_{}.txt'.format(station_name + file_suffix)))
            y_pred_vy = np.loadtxt(os.path.join(dir_vy, str(i_run), 'y_pred_{}.txt'.format(station_name + file_suffix)))
            y_pred_dir = vxy_to_dir_vec(y_pred_vx, y_pred_vy)

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
    n_runs = DefaultConfig.n_runs
    obs_data_path_list = DefaultConfig.obs_data_path_list
    station_name_list = DefaultConfig.station_name_list

    target = 'DIR'
    dir_in = os.path.join(DIR_LOG, file_exp_in)
    dir_log_target = os.path.join(dir_in, target)
    make_dir(dir_log_target)

    if mode.startswith('run'):
        data_generator_list = []
        for obs_data_path in obs_data_path_list:
            data_generator = DataGenerator(period, window, path=obs_data_path)
            data_generator_list.append(data_generator)

        for wid in range(TESTING_SLIDING_WINDOW, len(MONTH_LIST)):
            dir_log_exp = os.path.join(dir_log_target, str(MONTH_LIST[wid]))
            months = get_month_list(eval_mode, wid)
            for data_generator in data_generator_list:
                data_generator.set_data(months)
                data_generator.prepare_data(target_size,
                                            train_step=train_step, test_step=test_step, single_step=single_step)
            dir_vx = os.path.join(dir_in, 'VX', str(MONTH_LIST[wid]))
            dir_vy = os.path.join(dir_in, 'VY', str(MONTH_LIST[wid]))

            for tag_file in tag_file_list:
                run_dir_from_vxy(data_generator_list, dir_vx, dir_vy, dir_log_exp, tag_file, n_runs=n_runs)

    elif mode.startswith('reduce'):
        csv_result_list = []
        for tag_file in tag_file_list:
            csv = 'metrics_model.csv' if tag_file is None else 'metrics_model_{}.csv'.format(tag_file)
            csv_result_list.append(csv)
            csv = 'metrics_nwp.csv' if tag_file is None else 'metrics_nwp_{}.csv'.format(tag_file)
            csv_result_list.append(csv)
        reduce(csv_result_list, target, dir_log_target, n_runs, station_name_list)












