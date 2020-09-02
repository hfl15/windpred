import os
import numpy as np

from windpred.utils.base import get_tf_keras
from windpred.utils.data_parser import DataGeneratorV2, DataGeneratorV2Spatial
from windpred.utils.base import DIR_LOG
from windpred.utils.model_base import BasePredictor, CombinerDense
from windpred.utils.model_base import MONTH_LIST, TESTING_SLIDING_WINDOW, get_month_list
from windpred.utils.model_base import batch_run, run, reduce

from windpred.mhstn.base import get_data_temporal, get_data_spatial, get_evaluation_data_spatial, run_spatial
from windpred.utils.evaluation import Evaluator

tf, K = get_tf_keras()


class TemporalModule(BasePredictor):
    def __init__(self, input_shape, units_output=24, verbose=1, name='temporal_module'):
        super(TemporalModule, self).__init__(input_shape, units_output, verbose, name)
        self.input_shape_h = input_shape[0]
        self.input_shape_f = input_shape[1]
        self.model = self.build_model()
        if self.verbose > 0:
            self.model.summary()

    def build_model(self):
        input_h = tf.keras.layers.Input(self.input_shape_h)
        h_h = tf.keras.layers.LSTM(32)(input_h)
        h_h = tf.keras.layers.Dense(h_h.shape[1] * 2, activation='relu')(h_h)
        input_f = tf.keras.layers.Input(self.input_shape_f)
        h_f = tf.keras.layers.Flatten()(input_f)
        h_f = tf.keras.layers.Dense(h_f.shape[1] * 2, activation='relu')(h_f)
        h = tf.keras.layers.Concatenate()([h_h, h_f])
        h = tf.keras.layers.Dense(h.shape[1], activation='relu')(h)
        output = tf.keras.layers.Dense(self.units_output)(h)
        model = tf.keras.models.Model([input_h, input_f], output)
        return model


class SpatialModuleMLP(BasePredictor):
    def __init__(self, input_shape, units_output=24, verbose=1, name='spatial_module_mlp'):
        super(SpatialModuleMLP, self).__init__(input_shape, units_output, verbose, name)
        self.model = self.build_model()
        if self.verbose > 0:
            self.model.summary()

    def build_model(self):
        inp = tf.keras.layers.Input(self.input_shape)
        h = tf.keras.layers.Dense(inp.shape[1], activation='relu')(inp)
        output = tf.keras.layers.Dense(self.units_output)(h)
        model = tf.keras.models.Model(inp, output)
        return model


class SpatialModuleCNN(BasePredictor):
    def __init__(self, input_shape, units_output=24, verbose=1, name='spatial_module_cnn'):
        super(SpatialModuleCNN, self).__init__(input_shape, units_output, verbose, name)
        self.model = self.build_model()
        if self.verbose > 0:
            self.model.summary()

    def build_model(self):
        inp = tf.keras.layers.Input(self.input_shape)
        h = tf.keras.layers.Conv1D(64, 5, activation='relu')(inp)  # inp.shape[1]*n_stations*2
        h = tf.keras.layers.MaxPool1D()(h)
        h = tf.keras.layers.Flatten()(h)
        output = tf.keras.layers.Dense(self.units_output)(h)
        model = tf.keras.models.Model(inp, output)
        return model


class SpatialModuleCNNFull(BasePredictor):
    def __init__(self, input_shape, units_output=24, verbose=1, name='spatial_module_cnn_full'):
        super(SpatialModuleCNNFull, self).__init__(input_shape, units_output, verbose, name)
        self.model = self.build_model()
        if self.verbose > 0:
            self.model.summary()

    def build_model(self):
        inp = tf.keras.layers.Input(self.input_shape)
        n_channels = inp.shape[2]
        h = tf.keras.layers.Conv1D(inp.shape[1]*n_channels*2, inp.shape[1], activation='relu')(inp)
        h = tf.keras.layers.Flatten()(h)
        output = tf.keras.layers.Dense(self.units_output)(h)
        model = tf.keras.models.Model(inp, output)
        return model


def temporal_module(data_generator_list, dir_log, target, n_epochs,
                    features_history_in, features_future_in,
                    save_model=True):
    tag_func = temporal_module.__name__

    x_train_list, x_val_list, x_test_list, y_train_list, y_val_list, y_test_list = get_data_temporal(
        data_generator_list, target, features_history_in, features_future_in)
    input_shape = [x_train_list[0][0].shape[1:], x_train_list[0][1].shape[1:]]
    run(data_generator_list, TemporalModule, dir_log, target, n_epochs,
        x_train_list, x_val_list, x_test_list, y_train_list, y_val_list, y_test_list, input_shape, tag_func, save_model)


def get_data_spatial_output(station_name_list, dir_log, data_generator_spatial, target,
                            features_history, features_future, tag_temporal):
    x_train_list, x_val_list, x_test_list, y_train_list, y_val_list, y_test_list = get_data_spatial(
        data_generator_spatial, station_name_list, target, features_history, features_future)

    y_pred_train_list = []
    y_pred_val_list = []
    y_pred_test_list = []
    for i_station, station_name in enumerate(station_name_list):
        model = tf.keras.models.load_model(
            os.path.join(dir_log, '{}_{}.hdf5'.format(station_name, tag_temporal)))
        y_pred_train = model.predict(x_train_list[i_station])
        y_pred_val = model.predict(x_val_list[i_station])
        y_pred_test = model.predict(x_test_list[i_station])
        y_pred_train_list.append(y_pred_train)
        y_pred_val_list.append(y_pred_val)
        y_pred_test_list.append(y_pred_test)

    x_train = np.hstack(y_pred_train_list)
    x_val = np.hstack(y_pred_val_list)
    x_test = np.hstack(y_pred_test_list)
    input_shape = x_train.shape[1:]

    return x_train, x_val, x_test, y_train_list, y_val_list, y_test_list, input_shape


def get_data_spatial_hidden(station_name_list, dir_log, data_generator_spatial, target,
                            features_history, features_future, tag_temporal):
    n_stations = len(station_name_list)
    x_train_list, x_val_list, x_test_list, y_train_list, y_val_list, y_test_list = get_data_spatial(
        data_generator_spatial, station_name_list, target, features_history, features_future)

    model_list = []
    for i_station, station_name in enumerate(station_name_list):
        model = tf.keras.models.load_model(
            os.path.join(dir_log, '{}_{}.hdf5'.format(station_name, tag_temporal)))
        model_list.append(model)

    hidden_train_list, hidden_val_list, hidden_test_list = [], [], []
    for i in range(n_stations):
        local_inputs = model_list[i].inputs
        output_last_hidden = model_list[i].layers[-2].output
        get_output_last_hidden = tf.keras.backend.function(local_inputs, [output_last_hidden])
        hidden_train_list.append(get_output_last_hidden(x_train_list[i])[0])
        hidden_val_list.append(get_output_last_hidden(x_val_list[i])[0])
        hidden_test_list.append(get_output_last_hidden([x_test_list[i]])[0])

    return hidden_train_list, hidden_val_list, hidden_test_list, y_train_list, y_val_list, y_test_list


def get_data_spatial_mlp(station_name_list, dir_log, data_generator, target,
                         features_history, features_future, tag_temporal):
    hidden_train_list, hidden_val_list, hidden_test_list, y_train_list, y_val_list, y_test_list = \
        get_data_spatial_hidden(station_name_list, dir_log, data_generator, target, features_history, features_future,
                                tag_temporal)
    x_train = np.hstack(hidden_train_list)
    x_val = np.hstack(hidden_val_list)
    x_test = np.hstack(hidden_test_list)
    input_shape = x_train.shape[1:]
    return x_train, x_val, x_test, y_train_list, y_val_list, y_test_list, input_shape


def get_data_spatial_conv(station_name_list, dir_log, data_generator, target,
                          features_history, features_future, tag_temporal):

    hidden_train_list, hidden_val_list, hidden_test_list, y_train_list, y_val_list, y_test_list = \
        get_data_spatial_hidden(station_name_list, dir_log, data_generator, target, features_history, features_future,
                                tag_temporal)

    def concat(val_list):
        ret = val_list.copy()
        for i in range(len(ret)):
            ret[i] = ret[i].reshape(ret[i].shape+(1,))
        return np.concatenate(ret, axis=-1)
    x_train = concat(hidden_train_list)
    x_val = concat(hidden_val_list)
    x_test = concat(hidden_test_list)
    input_shape = x_train.shape[1:]

    return x_train, x_val, x_test, y_train_list, y_val_list, y_test_list, input_shape


def spatial_module(mode, station_name_list, dir_log, data_generator, target, n_epochs,
                   features_history, features_future, save_model=False):
    tag_func = spatial_module.__name__ + '_' + mode
    tag_temporal = temporal_module.__name__

    if mode == 'output':
        cls_model = CombinerDense
        get_data_func = get_data_spatial_output
    elif mode == 'mlp':
        cls_model = SpatialModuleMLP
        get_data_func = get_data_spatial_mlp
    elif mode == 'conv':
        cls_model = SpatialModuleCNN
        get_data_func = get_data_spatial_conv
    elif mode == 'conv_full':
        cls_model = SpatialModuleCNNFull
        get_data_func = get_data_spatial_conv
    else:
        raise ValueError("In {}: mode = {} can not be found!".format(spatial_module.__name__, mode))
    x_train, x_val, x_test, y_train_list, y_val_list, y_test_list, input_shape = get_data_func(
        station_name_list, dir_log, data_generator, target, features_history, features_future, tag_temporal)

    run_spatial(station_name_list, cls_model, dir_log, data_generator, target, n_epochs,
                x_train, x_val, x_test, y_train_list, y_val_list, y_test_list, input_shape, tag_func, save_model)


def combine_module(tag_spatial_suffix):
    def _combine_module(station_name_list, dir_log, data_generator, target, n_epochs,
                        features_history, features_future, save_model=False):
        tag_temporal = temporal_module.__name__
        tag_spatial = spatial_module.__name__ + '_' + tag_spatial_suffix
        tag_func = '{}_{}'.format(combine_module.__name__, tag_spatial_suffix)
        n_stations = len(station_name_list)

        _, _, _, y_train_list, y_val_list, y_test_list = get_data_spatial(
            data_generator, station_name_list, target, features_history, features_future)
        nwp, obs_list, speed_list, filter_big_wind_list = get_evaluation_data_spatial(data_generator, target,
                                                                                      n_stations)

        y_pred_temporal_train_list = []
        y_pred_temporal_val_list = []
        y_pred_temporal_test_list = []
        for i_station, station_name in enumerate(station_name_list):
            y_pred_tr = np.loadtxt(os.path.join(dir_log, 'y_pred_train_{}_{}.txt'.format(station_name, tag_temporal)))
            y_pred_val = np.loadtxt(os.path.join(dir_log, 'y_pred_val_{}_{}.txt'.format(station_name, tag_temporal)))
            y_pred_te = np.loadtxt(os.path.join(dir_log, 'y_pred_test_{}_{}.txt'.format(station_name, tag_temporal)))
            y_pred_temporal_train_list.append(y_pred_tr)
            y_pred_temporal_val_list.append(y_pred_val)
            y_pred_temporal_test_list.append(y_pred_te)

        y_pred_spatial_train_list = []
        y_pred_spatial_val_list = []
        y_pred_spatial_test_list = []
        for i_station, station_name in enumerate(station_name_list):
            y_pred_tr = np.loadtxt(os.path.join(dir_log, 'y_pred_train_{}_{}.txt'.format(station_name, tag_spatial)))
            y_pred_val = np.loadtxt(os.path.join(dir_log, 'y_pred_val_{}_{}.txt'.format(station_name, tag_spatial)))
            y_pred_te = np.loadtxt(os.path.join(dir_log, 'y_pred_test_{}_{}.txt'.format(station_name, tag_spatial)))
            y_pred_spatial_train_list.append(y_pred_tr)
            y_pred_spatial_val_list.append(y_pred_val)
            y_pred_spatial_test_list.append(y_pred_te)

        evaluator_model = Evaluator(dir_log, 'model_{}'.format(tag_func))
        evaluator_nwp = Evaluator(dir_log, 'nwp_{}'.format(tag_func))
        for i_station in range(n_stations):
            station_name = station_name_list[i_station]
            y_train, y_val, y_test = y_train_list[i_station], y_val_list[i_station], y_test_list[i_station]
            y_com_train = np.hstack([y_pred_temporal_train_list[i_station], y_pred_spatial_train_list[i_station]])
            y_com_val = np.hstack([y_pred_temporal_val_list[i_station], y_pred_spatial_val_list[i_station]])
            y_com_test = np.hstack([y_pred_temporal_test_list[i_station], y_pred_spatial_test_list[i_station]])

            combine_model = CombinerDense(y_com_train.shape[1:], name='{}_{}'.format(station_name, tag_func))
            combine_model.fit(y_com_train, y_train, n_epochs=n_epochs, validation_data=(y_com_val, y_val))

            y_pred = combine_model.predict(y_com_test).ravel()
            if data_generator.norm is not None:
                target_curr = '{}_S{}'.format(target, i_station)
                y_pred = data_generator.normalizer.inverse_transform(target_curr, y_pred)

            obs = obs_list[i_station]
            filter_big_wind = filter_big_wind_list[i_station]
            evaluator_model.append(obs, y_pred, filter_big_wind, key=station_name)
            evaluator_nwp.append(obs, nwp, filter_big_wind, key=station_name)

            if save_model:
                combine_model.save(dir_log)
            np.savetxt(os.path.join(dir_log, 'y_pred_{}_{}.txt'.format(station_name, tag_func)), y_pred)
            np.savetxt(os.path.join(dir_log, 'y_pred_train_{}_{}.txt'.format(station_name, tag_func)),
                       combine_model.predict(y_com_train))
            np.savetxt(os.path.join(dir_log, 'y_pred_val_{}_{}.txt'.format(station_name, tag_func)),
                       combine_model.predict(y_com_val))
            np.savetxt(os.path.join(dir_log, 'y_pred_test_{}_{}.txt'.format(station_name, tag_func)),
                       combine_model.predict(y_com_test))
    return _combine_module


def get_tags(mode='all'):
    tag_func_temporal = [temporal_module.__name__]
    spatial_mode_list = ['output', 'mlp', 'conv', 'conv_full']
    tag_func_spatial_list = [spatial_module.__name__ + '_' + suffix for suffix
                             in spatial_mode_list]
    tag_func_combine_list = [combine_module.__name__ + '_' + suffix for suffix
                             in spatial_mode_list]
    tag_func_list = tag_func_temporal + tag_func_spatial_list + tag_func_combine_list

    if mode == 'temporal':
        return tag_func_temporal
    elif mode == 'spatial':
        return tag_func_spatial_list
    elif mode == 'combine':
        return tag_func_combine_list
    elif mode == 'all':
        return tag_func_list
    else:
        raise ValueError("mode = {} can not be found!".format(mode))


def main(target, mode, eval_mode, config, tag, features_history, features_future, csv_result_list=None):
    target_size = config.target_size
    period = config.period
    window = config.window
    train_step = config.train_step
    test_step = config.test_step
    single_step = config.single_step
    norm = config.norm
    x_divide_std = config.x_divide_std
    n_epochs = config.n_epochs
    n_runs = config.n_runs
    obs_data_path_list = config.obs_data_path_list
    station_name_list = config.station_name_list

    dir_log_target = os.path.join(DIR_LOG, tag, target)

    if mode.startswith('temporal'):
        data_generator_list = []
        for obs_data_path in obs_data_path_list:
            data_generator = DataGeneratorV2(period, window, path=obs_data_path, norm=norm, x_divide_std=x_divide_std)
            data_generator_list.append(data_generator)

        for wid in range(TESTING_SLIDING_WINDOW, len(MONTH_LIST)):
            dir_log_exp = os.path.join(dir_log_target, str(MONTH_LIST[wid]))
            months = get_month_list(eval_mode, wid)
            for data_generator in data_generator_list:
                data_generator.set_data(months)
                data_generator.prepare_data(target_size, train_step=train_step, test_step=test_step,
                                            single_step=single_step)
            batch_run(n_runs, dir_log_exp,
                      lambda dir_log_curr: temporal_module(data_generator_list, dir_log_curr, target, n_epochs,
                                                           features_history, features_future))
    elif mode.startswith('spatial'):
        data_generator_spatial = DataGeneratorV2Spatial(period, window, norm=norm, x_divide_std=x_divide_std)
        for wid in range(TESTING_SLIDING_WINDOW, len(MONTH_LIST)):
            dir_log_exp = os.path.join(dir_log_target, str(MONTH_LIST[wid]))
            months = get_month_list(eval_mode, wid)
            data_generator_spatial.set_data(months)
            data_generator_spatial.prepare_data(target_size,
                                                train_step=train_step, test_step=test_step, single_step=single_step)
            batch_run(n_runs, dir_log_exp,
                      lambda dir_log_curr: spatial_module(mode.split('-')[-1], station_name_list, dir_log_curr,
                                                          data_generator_spatial, target, n_epochs,
                                                          features_history, features_future))
    elif mode.startswith('combine'):
        data_generator_spatial = DataGeneratorV2Spatial(period, window, norm=norm, x_divide_std=x_divide_std)
        for wid in range(TESTING_SLIDING_WINDOW, len(MONTH_LIST)):
            dir_log_exp = os.path.join(dir_log_target, str(MONTH_LIST[wid]))
            months = get_month_list(eval_mode, wid)
            data_generator_spatial.set_data(months)
            data_generator_spatial.prepare_data(target_size,
                                                train_step=train_step, test_step=test_step, single_step=single_step)
            combine_module_func = combine_module(mode.split('-')[-1])
            batch_run(n_runs, dir_log_exp,
                      lambda dir_log_curr: combine_module_func(
                          station_name_list, dir_log_curr, data_generator_spatial, target, n_epochs, features_history,
                          features_future))
    elif mode.startswith('reduce'):
        reduce(csv_result_list, target, dir_log_target, n_runs, station_name_list)

    else:
        raise ValueError("mode = {} can not be found!".format(mode))





