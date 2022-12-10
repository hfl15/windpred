import os

from windpred.utils.base import DIR_LOG
from windpred.utils.base import make_dir
from windpred.utils.data_parser import DataGenerator
from windpred.utils.model_base import MONTH_LIST, TESTING_SLIDING_WINDOW, get_month_list, DefaultConfig
from windpred.utils.model_base import batch_run, run, reduce


def main(tag, config: DefaultConfig, target, mode, eval_mode, cls_model, csv_result_list=None):
    dir_log_mode = os.path.join(DIR_LOG, tag, mode.split('-')[-1])
    dir_log_target = os.path.join(dir_log_mode, target)
    make_dir(dir_log_target)

    if mode.startswith('run'):
        data_generator_list = []
        for obs_data_path in config.obs_data_path_list:
            data_generator = DataGenerator(config.period, config.window, path=obs_data_path)
            data_generator_list.append(data_generator)

        for wid in range(TESTING_SLIDING_WINDOW, len(MONTH_LIST)):
            dir_log_exp = os.path.join(dir_log_target, str(MONTH_LIST[wid]))
            months = get_month_list(eval_mode, wid)
            for data_generator in data_generator_list:
                data_generator.set_data(months)
                data_generator.prepare_data(config.target_size, train_step=config.train_step,
                                            test_step=config.test_step, single_step=config.single_step)

            if mode == 'run-history':
                features = [target]
            elif mode == 'run-future':
                features = ['NEXT_NWP_{}'.format(target)]
            elif mode == 'run-history_future':
                features = [target, 'NEXT_NWP_{}'.format(target)]
            else:
                raise ValueError('mode={} can not be found!'.format(mode))
            x_train_list, x_val_list, x_test_list = [], [], []
            y_train_list, y_val_list, y_test_list = [], [], []
            for data_generator in data_generator_list:
                (x_train, y_train), (x_val, y_val), (x_test, y_test) = \
                    data_generator.extract_training_data(x_attributes=features, y_attributes=[target])
                x_train_list.append(x_train)
                x_val_list.append(x_val)
                x_test_list.append(x_test)
                y_train_list.append(y_train)
                y_val_list.append(y_val)
                y_test_list.append(y_test)
            input_shape = x_train_list[0].shape[1:]

            batch_run(config.n_runs, dir_log_exp,
                      lambda dir_log_curr:
                      run(data_generator_list, cls_model, dir_log_curr, target, config.n_epochs,
                          x_train_list, x_val_list, x_test_list,
                          y_train_list, y_val_list, y_test_list, input_shape))

    elif mode.startswith('reduce'):
        if csv_result_list is None:
            csv_result_list = ['metrics_model.csv', 'metrics_nwp.csv']
        reduce(csv_result_list, dir_log_target, config.n_runs, config.station_name_list)