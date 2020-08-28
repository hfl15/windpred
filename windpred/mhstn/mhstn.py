import os

from windpred.utils.base import get_tf_keras
from windpred.utils.data_parser import DataGeneratorV2
from windpred.utils.base import DIR_LOG
from windpred.utils.model_base import BasePredictor
from windpred.utils.model_base import MONTH_LIST, TESTING_SLIDING_WINDOW, get_month_list
from windpred.utils.model_base import batch_run, run, reduce

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


def temporal_module(data_generator_list, dir_log, target, n_epochs,
                    features_history_in, features_future_in,
                    save_model=True):
    tag_func = temporal_module.__name__
    x_train_list, x_val_list, x_test_list = [], [], []
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
    input_shape = [x_train_list[0][0].shape[1:], x_train_list[0][1].shape[1:]]
    run(data_generator_list, TemporalModule, dir_log, target, n_epochs,
        x_train_list, x_val_list, x_test_list, y_train_list, y_val_list, y_test_list, input_shape, tag_func,save_model)


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
    elif mode.startswith('reduce'):
        reduce(csv_result_list, target, dir_log_target, n_runs, station_name_list)





