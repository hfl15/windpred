"""
    Realizations of conventional models.
"""

import os
import numpy as np

from windpred.utils.base import DIR_LOG
from windpred.utils.base import make_dir
from windpred.utils.data_parser import DataGenerator
from windpred.utils.model_base import MONTH_LIST, TESTING_SLIDING_WINDOW, get_month_list
from windpred.utils.model_base import batch_run, reduce, DefaultConfig

from windpred.utils.base import tag_path
from windpred.expinc.base import eval_mode

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV


def run(data_generator_list, cls_model, dir_log, target, n_epochs,
        x_train_list, x_val_list, x_test_list,
        y_train_list, y_val_list, y_test_list,
        input_shape, tag_file=None, save_model=False):
    file_suffix = "" if tag_file is None else '_'+tag_file
    evaluator_model = Evaluator(dir_log, 'model'+file_suffix)
    evaluator_nwp = Evaluator(dir_log, 'nwp'+file_suffix)
    time_training, time_inference = 0, 0
    for i_station, data_generator in enumerate(data_generator_list):
        time_start = time()
        station_name = data_generator.station_name
        x_train, x_val, x_test = x_train_list[i_station], x_val_list[i_station], x_test_list[i_station]
        y_train, y_val, y_test = y_train_list[i_station], y_val_list[i_station], y_test_list[i_station]

        model = cls_model(input_shape, name=(station_name+file_suffix))
        model.fit(x_train, y_train, n_epochs=n_epochs, validation_data=(x_val, y_val))
        time_training += (time() - time_start)
        model.plot_train_history(dir_log, ('train_loss'+'_'+station_name+file_suffix))

        time_start = time()
        y_pred = model.predict(x_test).ravel()
        if data_generator.norm is not None:
            y_pred = data_generator.normalizer.inverse_transform(target, y_pred)
        time_inference += (time() - time_start)
        speed, nwp, obs, filter_big_wind = data_generator.extract_evaluation_data(target)
        plot_and_save_comparison(obs, y_pred, dir_log, filename='compare_{}.png'.format(station_name+file_suffix))
        evaluator_model.append(obs, y_pred, filter_big_wind, key=station_name)
        evaluator_nwp.append(obs, nwp, filter_big_wind, key=station_name)

        if save_model:
            model.save(dir_log)
        np.savetxt(os.path.join(dir_log, 'y_pred_{}.txt'.format(station_name+file_suffix)), y_pred)

    return time_training, time_inference


def main(tag, config: DefaultConfig, target, mode, eval_mode, csv_result_list=None):
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
                data_generator.prepare_data(config.target_size,
                                            train_step=config.train_step, test_step=config.test_step,
                                            single_step=config.single_step)

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

            from IPython import embed; embed()

            ##### the code below passed test.
            i_station = 0
            x_train, x_val, x_test = x_train_list[0], x_val_list[0], x_test_list[0]
            y_train, y_val, y_test = y_train_list[0], y_val_list[0], y_test_list[0]

            x_train = x_train.reshape([x_train.shape[0], -1])
            x_val = x_val.reshape([x_val.shape[0], -1])
            x_test = x_test.reshape([x_test.shape[0], -1])

            n_samples_tr = x_train.shape[0]
            n_samples_val = x_val.shape[0]
            x = np.vstack([x_train, x_val])
            y = np.vstack([y_train, y_val])
            train_ids = np.arange(0, n_samples_tr)
            val_ids = np.arange(n_samples_tr, n_samples_tr+n_samples_val)

            params = {
                'estimator__alpha': (0.1, 0.3, 0.5, 0.7, 0.9),
                'estimator__l1_ratio': (0.1, 0.3, 0.5, 0.7, 0.9)
            }  # parameters in a pipeline, refer to (https://scikit-learn.org/stable/modules/compose.html#pipeline) for more detail.

            model = MultiOutputRegressor(ElasticNet(normalize=False, precompute=False))
            finder = GridSearchCV(
                estimator=model,
                param_grid=params,
                verbose=1,
                pre_dispatch=8,
                error_score=-999,
                return_train_score=True,
                refit=False,   # our DNN models did not refit.
                cv=[(train_ids, val_ids)]
            )
            finder.fit(x, y)
            ### ------------------------ above




            params = {
                'estimator__n_estimators': [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
            }
            model = MultiOutputRegressor(GradientBoostingRegressor())

            model = MultiOutputRegressor(GradientBoostingRegressor(n_estimators=5))

            model.fit(x_train, y_train)
            y_pred = model.predict(x_val)

            batch_run(config.n_runs, dir_log_exp,
                      lambda dir_log_curr:
                      run(data_generator_list, cls_model, dir_log_curr, target, config.n_epochs,
                          x_train_list, x_val_list, x_test_list,
                          y_train_list, y_val_list, y_test_list, input_shape))

    elif mode.startswith('reduce'):
        if csv_result_list is None:
            csv_result_list = ['metrics_model.csv', 'metrics_nwp.csv']
        reduce(csv_result_list, target, dir_log_target, config.n_runs, config.station_name_list)


if __name__ == '__main__':
    tag = tag_path(os.path.abspath(__file__), 2)

    mode = 'run'
    target = 'V'
    feature_mode_list = ['history', 'future', 'history_future']
    feature_mode = feature_mode_list[-1]
    main(tag, DefaultConfig(), target, mode + '-' + feature_mode, eval_mode)


