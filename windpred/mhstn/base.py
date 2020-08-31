import os
import numpy as np

from windpred.utils.evaluation import Evaluator


def get_data_spatial(data_generator, station_name_list, target, features_history_in, features_future_in):
    def _get_station(station_idx, features_history, features_future):
        y_attributes = ['{}_S{}'.format(target, station_idx)]
        x_attributes = ['{}_S{}'.format(f, station_idx) for f in features_history] + features_future
        (x_train, y_train), (x_val, y_val), (x_test, y_test) = \
            data_generator.extract_training_data(
                x_attributes=x_attributes, y_attributes=y_attributes)

        xh_train, xf_train = x_train[:, :, :len(features_history)], x_train[:, :, len(features_history):]
        xh_val, xf_val = x_val[:, :, :len(features_history)], x_val[:, :, len(features_history):]
        xh_test, xf_test = x_test[:, :, :len(features_history)], x_test[:, :, len(features_history):]
        x_train = [xh_train, xf_train]
        x_val = [xh_val, xf_val]
        x_test = [xh_test, xf_test]

        return (x_train, y_train), (x_val, y_val), (x_test, y_test)

    x_train_list, x_val_list, x_test_list = [], [], []  # [[[history data], [future data]],...]
    y_train_list, y_val_list, y_test_list = [], [], []
    for i, station_name in enumerate(station_name_list):

        if type(features_history_in) == dict:
            features_history = features_history_in[station_name]
        else:
            features_history = features_history_in
        if type(features_future_in) == dict:
            features_future = features_future_in[station_name]
        else:
            features_future = features_future_in

        (x_train, y_train), (x_val, y_val), (x_test, y_test) = _get_station(i, features_history, features_future)
        x_train_list.append(x_train)
        x_val_list.append(x_val)
        x_test_list.append(x_test)
        y_train_list.append(y_train)
        y_val_list.append(y_val)
        y_test_list.append(y_test)

    return x_train_list, x_val_list, x_test_list, y_train_list, y_val_list, y_test_list


def get_data_temporal(data_generator_list, target, features_history_in, features_future_in):
    x_train_list, x_val_list, x_test_list = [], [], []  # [[[history data], [future data]],...]
    y_train_list, y_val_list, y_test_list = [], [], []
    for data_generator in data_generator_list:
        station_name = data_generator.station_name

        if type(features_history_in) == dict:
            features_history = features_history_in[station_name]
        else:
            features_history = features_history_in
        if type(features_future_in) == dict:
            features_future = features_future_in[station_name]
        else:
            features_future = features_future_in

        (x_train, y_train), (x_val, y_val), (x_test, y_test) = \
            data_generator.extract_training_data(x_attributes=features_history+features_future,
                                                 y_attributes=[target])
        xh_train, xf_train = x_train[:, :, :len(features_history)], x_train[:, :, len(features_history):]
        xh_val, xf_val = x_val[:, :, :len(features_history)], x_val[:, :, len(features_history):]
        xh_test, xf_test = x_test[:, :, :len(features_history)], x_test[:, :, len(features_history):]
        x_train = [xh_train, xf_train]
        x_val = [xh_val, xf_val]
        x_test = [xh_test, xf_test]

        x_train_list.append(x_train)
        x_val_list.append(x_val)
        x_test_list.append(x_test)
        y_train_list.append(y_train)
        y_val_list.append(y_val)
        y_test_list.append(y_test)

    return x_train_list, x_val_list, x_test_list, y_train_list, y_val_list, y_test_list


def get_evaluation_data_spatial(data_generator, target, n_stations):
    nwp, obs_list, speed_list, filter_big_wind_list = None, [], [], []
    for i in range(n_stations):
        target_i = '{}_S{}'.format(target, i)
        speed, nwp, obs, filter_big_wind = data_generator.extract_evaluation_data(target_i)
        obs_list.append(obs)
        speed_list.append(speed)
        filter_big_wind_list.append(filter_big_wind)

    return nwp, obs_list, speed_list, filter_big_wind_list


def run_spatial(station_name_list, cls_model, dir_log, data_generator, target, n_epochs,
                x_train, x_val, x_test, y_train_list, y_val_list, y_test_list, input_shape,
                tag_file=None, save_model=False):

    tag_func = "" if tag_file is None else '_' + tag_file
    n_stations = len(station_name_list)
    nwp, obs_list, speed_list, filter_big_wind_list = get_evaluation_data_spatial(data_generator, target, n_stations)

    evaluator_model = Evaluator(dir_log, 'model'+tag_func)
    evaluator_nwp = Evaluator(dir_log, 'nwp{}'+tag_func)
    for i_station in range(n_stations):
        station_name = station_name_list[i_station]
        y_train, y_val, y_test = y_train_list[i_station], y_val_list[i_station], y_test_list[i_station]

        model = cls_model(input_shape, name='{}_{}'.format(station_name, tag_func))
        model.fit(x_train, y_train, n_epochs=n_epochs, validation_data=(x_val, y_val))

        y_pred = model.predict(x_test).ravel()
        if data_generator.norm is not None:
            target_curr = '{}_S{}'.format(target, i_station)
            y_pred = data_generator.normalizer.inverse_transform(target_curr, y_pred)

        obs = obs_list[i_station]
        filter_big_wind = filter_big_wind_list[i_station]
        evaluator_model.append(obs, y_pred, filter_big_wind, key=station_name)
        evaluator_nwp.append(obs, nwp, filter_big_wind, key=station_name)

        if save_model:
            model.save(dir_log)
        file_suffix = '{}_{}'.format(station_name, tag_func)
        np.savetxt(os.path.join(dir_log, 'y_pred_{}.txt'.format(file_suffix)), y_pred)
        np.savetxt(os.path.join(dir_log, 'y_pred_train_{}.txt'.format(file_suffix)), model.predict(x_train))
        np.savetxt(os.path.join(dir_log, 'y_pred_val_{}.txt'.format(file_suffix)), model.predict(x_val))
        np.savetxt(os.path.join(dir_log, 'y_pred_test_{}.txt'.format(file_suffix)), model.predict(x_test))