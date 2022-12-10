"""
    Implementations of conventional models.
"""

import os
import numpy as np
from time import time

from windpred.utils.base import DIR_LOG, plot_and_save_comparison
from windpred.utils.base import make_dir
from windpred.utils.data_parser import DataGenerator
from windpred.utils.evaluation import Evaluator
from windpred.utils.model_base import MONTH_LIST, TESTING_SLIDING_WINDOW, get_month_list
from windpred.utils.model_base import batch_run, reduce, DefaultConfig

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import RandomizedSearchCV

MODELS = {
    'gbrt': [GradientBoostingRegressor(),
             {'learning_rate': [0.001, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5],
              'n_estimators': [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
              'alpha': [0.1, 0.3, 0.6, 0.9],
              'n_iter_no_change': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]}],
    'svr': [SVR(),
            {'degree': [3, 4, 5, 6, 7, 8, 9, 10],
             'C': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
             'epsilon': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]}]
}


def run(data_generator_list, model_name, dir_log, target,
        x_train_list, x_val_list, x_test_list,
        y_train_list, y_val_list, y_test_list,
        save_model=False):
    evaluator_model = Evaluator(dir_log, 'model')
    evaluator_nwp = Evaluator(dir_log, 'nwp')
    time_training, time_inference = 0, 0
    for i_station, data_generator in enumerate(data_generator_list):
        time_start = time()
        station_name = data_generator.station_name
        x_train, x_val, x_test = x_train_list[i_station], x_val_list[i_station], x_test_list[i_station]
        y_train, y_val, y_test = y_train_list[i_station], y_val_list[i_station], y_test_list[i_station]

        x_train = x_train.reshape([x_train.shape[0], -1])
        x_val = x_val.reshape([x_val.shape[0], -1])
        x_test = x_test.reshape([x_test.shape[0], -1])
        n_samples_tr = x_train.shape[0]
        n_samples_val = x_val.shape[0]
        x = np.vstack([x_train, x_val])
        y = np.vstack([y_train, y_val])
        train_ids = np.arange(0, n_samples_tr)
        val_ids = np.arange(n_samples_tr, n_samples_tr + n_samples_val)

        model, params = MODELS[model_name]
        params_wrapper = {}
        for key, val in params.items():
            params_wrapper['estimator__' + key] = val
        model = MultiOutputRegressor(model)
        finder = RandomizedSearchCV(estimator=model, param_distributions=params_wrapper, n_iter=30, refit=True,
                                    verbose=1, cv=[(train_ids, val_ids)])
        finder.fit(x, y)
        time_training += (time() - time_start)

        # # useless code segment, but it may be useful in the future to alleviate the following case:
        # # the setting `refit=True` in RandomizedSearchCV will lead the program to refit an estimator using the best
        # # found parameters on the whole dataset, including both training and validation set. Comparing such a model to
        # # a DNN based model may be unfair because the DNN model is just trained on the training set.
        # best_params = dict()
        # for key, value in finder.best_params_.items():
        #     best_params[key.split('__')[-1]] = value
        # model_cls, _ = MODELS[model_name]
        # model = model_cls(**best_params)
        # model = MultiOutputRegressor(model)
        # model.fit(x_train, y_train)
        # y_pred = model.predict(x_test).ravel()

        time_start = time()
        y_pred = finder.predict(x_test).ravel()
        if data_generator.norm is not None:
            y_pred = data_generator.normalizer.inverse_transform(target, y_pred)
        time_inference += (time() - time_start)

        speed, nwp, obs, filter_big_wind = data_generator.extract_evaluation_data(target)
        plot_and_save_comparison(obs, y_pred, dir_log, filename='compare_{}.png'.format(station_name))
        evaluator_model.append(obs, y_pred, filter_big_wind, key=station_name)
        evaluator_nwp.append(obs, nwp, filter_big_wind, key=station_name)

        np.savetxt(os.path.join(dir_log, 'y_pred_{}.txt'.format(station_name)), y_pred)

    return time_training, time_inference


def run_spatial(data_generator_list, model_name, dir_log, target,
                x_train_list, x_val_list, x_test_list,
                y_train_list, y_val_list, y_test_list,
                save_model=False):
    # construct dataset x
    x_train_spatial, x_test_spatial = [], []
    for i_station, data_generator in enumerate(data_generator_list):
        x_train, x_val, x_test = x_train_list[i_station], x_val_list[i_station], x_test_list[i_station]

        x_train = x_train.reshape([x_train.shape[0], -1])
        x_val = x_val.reshape([x_val.shape[0], -1])
        x_test = x_test.reshape([x_test.shape[0], -1])
        x_train = np.vstack([x_train, x_val])
        x_train_spatial.append(x_train)
        x_test_spatial.append(x_test)
    x_train_spatial = np.hstack(x_train_spatial)
    x_test_spatial = np.hstack(x_test_spatial)

    # start to train
    evaluator_model = Evaluator(dir_log, 'model')
    evaluator_nwp = Evaluator(dir_log, 'nwp')
    time_training, time_inference = 0, 0
    for i_station, data_generator in enumerate(data_generator_list):
        time_start = time()
        station_name = data_generator.station_name

        y_train, y_val, y_test = y_train_list[i_station], y_val_list[i_station], y_test_list[i_station]
        n_samples_tr = y_train.shape[0]
        n_samples_val = y_val.shape[0]
        y_train = np.vstack([y_train, y_val])
        train_ids = np.arange(0, n_samples_tr)
        val_ids = np.arange(n_samples_tr, n_samples_tr + n_samples_val)

        model, params = MODELS[model_name]
        params_wrapper = {}
        for key, val in params.items():
            params_wrapper['estimator__' + key] = val
        model = MultiOutputRegressor(model)
        finder = RandomizedSearchCV(estimator=model, param_distributions=params_wrapper, n_iter=30, refit=True,
                                    verbose=1, cv=[(train_ids, val_ids)])
        finder.fit(x_train_spatial, y_train)
        time_training += (time() - time_start)

        time_start = time()
        y_pred = finder.predict(x_test_spatial).ravel()
        if data_generator.norm is not None:
            y_pred = data_generator.normalizer.inverse_transform(target, y_pred)
        time_inference += (time() - time_start)

        speed, nwp, obs, filter_big_wind = data_generator.extract_evaluation_data(target)
        plot_and_save_comparison(obs, y_pred, dir_log, filename='compare_{}.png'.format(station_name))
        evaluator_model.append(obs, y_pred, filter_big_wind, key=station_name)
        evaluator_nwp.append(obs, nwp, filter_big_wind, key=station_name)

        np.savetxt(os.path.join(dir_log, 'y_pred_{}.txt'.format(station_name)), y_pred)

    return time_training, time_inference


def main(tag, config: DefaultConfig, target, mode, eval_mode, model_name='gbrt', csv_result_list=None,
         features=None, with_spatial=False):
    dir_log_mode = make_dir(os.path.join(DIR_LOG, tag, model_name))
    dir_log_target = make_dir(os.path.join(dir_log_mode, target))
    if features is None:
        features = [target, 'NEXT_NWP_{}'.format(target)]

    if mode.startswith('run'):
        data_generator_list = []
        for obs_data_path in config.obs_data_path_list:
            data_generator = DataGenerator(config.period, config.window, path=obs_data_path)
            data_generator_list.append(data_generator)

        for wid in range(TESTING_SLIDING_WINDOW, len(MONTH_LIST)):
            dir_log_exp = make_dir(os.path.join(dir_log_target, str(MONTH_LIST[wid])))
            months = get_month_list(eval_mode, wid)
            for data_generator in data_generator_list:
                data_generator.set_data(months)
                data_generator.prepare_data(config.target_size,
                                            train_step=config.train_step, test_step=config.test_step,
                                            single_step=config.single_step)

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

            if with_spatial:
                run_func = run_spatial
            else:
                run_func = run
            batch_run(config.n_runs, dir_log_exp,
                      lambda dir_log_curr:
                      run_func(data_generator_list, model_name, dir_log_curr, target,
                               x_train_list, x_val_list, x_test_list,
                               y_train_list, y_val_list, y_test_list))

    elif mode.startswith('reduce'):
        if csv_result_list is None:
            csv_result_list = ['metrics_model.csv', 'metrics_nwp.csv']
        reduce(csv_result_list, dir_log_target, config.n_runs, config.station_name_list)





