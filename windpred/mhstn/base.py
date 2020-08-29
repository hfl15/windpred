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