import os
import numpy as np

from windpred.utils.evaluation import Evaluator, EvaluatorDir
from windpred.utils.base import make_dir, DIR_LOG
from windpred.utils.data_parser import DataGenerator
from windpred.utils.model_base import MONTH_LIST, TESTING_SLIDING_WINDOW, get_month_list, DefaultConfig, reduce


def eval(config:DefaultConfig, target, dir_in, dir_out, eval_mode, tag_file=None):
    data_generator_list = []
    for obs_data_path in config.obs_data_path_list:
        data_generator = DataGenerator(config.period, config.window, path=obs_data_path)
        data_generator_list.append(data_generator)

    for wid in range(TESTING_SLIDING_WINDOW, len(MONTH_LIST)):
        dir_in_date = os.path.join(dir_in, target, str(MONTH_LIST[wid]))
        dir_out_date = make_dir(os.path.join(dir_out, target, str(MONTH_LIST[wid])))

        months = get_month_list(eval_mode, wid)
        for data_generator in data_generator_list:
            data_generator.set_data(months)
            data_generator.prepare_data(config.target_size, train_step=config.train_step,
                                        test_step=config.test_step, single_step=config.single_step)

        for ir in range(config.n_runs):
            dir_in_run = os.path.join(dir_in_date, str(ir))
            dir_out_run = make_dir(os.path.join(dir_out_date, str(ir)))

            file_suffix = "" if tag_file is None else '_' + tag_file
            if target == "DIR":
                evaluator_model = EvaluatorDir(dir_out_run, 'model' + file_suffix)
                evaluator_nwp = EvaluatorDir(dir_out_run, 'nwp' + file_suffix)
            else:
                evaluator_model = Evaluator(dir_out_run, 'model' + file_suffix)
                evaluator_nwp = Evaluator(dir_out_run, 'nwp' + file_suffix)
            for i_station, data_generator in enumerate(data_generator_list):
                station_name = data_generator.station_name
                speed, nwp, obs, filter_big_wind = data_generator.extract_evaluation_data(target)
                y_pred_file = os.path.join(dir_in_run, "y_pred_test_{}.txt".format(station_name))
                y_pred = np.loadtxt(y_pred_file).ravel()
                evaluator_model.append(obs, y_pred, filter_big_wind, key=station_name)
                evaluator_nwp.append(obs, nwp, filter_big_wind, key=station_name)


if __name__ == '__main__':

    for target in ['V', 'VX', 'VY', 'DIR']:
        eval_mode = 'rolling'
        dir_in = os.path.join(DIR_LOG, 'exproll_base_duq_best')
        dir_out = make_dir('{}_eval'.format(dir_in))
        config = DefaultConfig()

        eval(config, target, dir_in, dir_out, eval_mode)

        csv_result_list = ['metrics_model.csv', 'metrics_nwp.csv']
        reduce(csv_result_list, os.path.join(dir_out, target), config.n_runs, config.station_name_list)




