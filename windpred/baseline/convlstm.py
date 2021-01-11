"""
    ConvolutionalLSTM.
"""
import tensorflow as tf
import numpy as np
import os

from windpred.utils.base import DIR_LOG, make_dir
from windpred.utils.data_parser import DataGeneratorSpatial
from windpred.utils.model_base import BasePredictor, batch_run
from windpred.utils.model_base import DefaultConfig
from windpred.utils.model_base import TESTING_SLIDING_WINDOW, MONTH_LIST, get_month_list, reduce
from windpred.mhstn.base import get_data_spatial, run_spatial


class ConvLSTM(BasePredictor):
    def __init__(self, input_shape, units_output=24, verbose=1, name='convlstm'):
        # input_shape = (seq_len, n_features, 1, n_channels)
        super(ConvLSTM, self).__init__(input_shape, units_output, verbose, name)
        self.model = self.build_model()
        if self.verbose > 0:
            self.model.summary()

    def build_model(self):
        model = tf.keras.Sequential(
            [
                tf.keras.layers.Input(
                    shape=self.input_shape
                ),
                tf.keras.layers.ConvLSTM2D(
                    filters=32, kernel_size=(5, 5), padding="same", return_sequences=True
                    # return_sequences=True is critical to obtain high performance.
                ),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(self.units_output)
            ]
        )

        return model

    def fit(self, x, y,
            batch_size=32,
            n_epochs=20,
            validation_data=None,
            set_cb=True):
        callbacks = None
        if set_cb:
            reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
                factor=0.5, patience=3, min_lr=0.0001, mode='min', verbose=self.verbose)
            es = tf.keras.callbacks.EarlyStopping(patience=20, mode='min', verbose=self.verbose,
                                                  restore_best_weights=True)
            callbacks = [reduce_lr, es]

        op = tf.keras.optimizers.Adam(learning_rate=0.001)
        self.model.compile(loss='mse', optimizer=op)
        self.history = self.model.fit(x, y,
                                      batch_size=batch_size,
                                      epochs=n_epochs,
                                      verbose=self.verbose,
                                      validation_data=validation_data,
                                      callbacks=callbacks)

        return self.history.history['loss'][-1], self.history.history['val_loss'][-1]


def run_convlstm(station_name_list, dir_log, data_generator_spatial, target, n_epochs,
                 features_history=None, features_future=None, model_name='convlstm'):
    tag_func = model_name

    x_train_list, x_val_list, x_test_list, y_train_list, y_val_list, y_test_list = get_data_spatial(
        data_generator_spatial, station_name_list, target, features_history, features_future)

    def get_data(x_list):
        res = []
        for x in x_list:
            if type(x) == list and len(x) == 2:
                x_h, x_f = x
                x_ = np.concatenate([x_h, x_f], axis=-1)
            else:
                x_ = x
            x_ = x_.reshape(x_.shape+(1,))  # (#samples, seq_len, n_features, 1, n_channels)
            res.append(x_)
        return np.stack(res, axis=-1)

    x_tr = get_data(x_train_list)
    x_val = get_data(x_val_list)
    x_te = get_data(x_test_list)
    seq_len = x_tr.shape[1]
    n_features = x_tr.shape[2]
    n_channels = x_tr.shape[-1]
    input_shape = (seq_len, n_features, 1, n_channels)

    if model_name == 'convlstm':
        cls_model = ConvLSTM
    else:
        raise ValueError("model_name={} can not be found!".format(model_name))

    run_spatial(station_name_list, cls_model, dir_log, data_generator_spatial, target, n_epochs,
                x_tr, x_val, x_te, y_train_list, y_val_list, y_test_list, input_shape, tag_func)


def main(target, mode, eval_mode, config: DefaultConfig, tag, model_name, features_history, features_future):
    # target_size = config.target_size
    # period = config.period
    # window = config.window
    # train_step = config.train_step
    # test_step = config.test_step
    # single_step = config.single_step
    # norm = config.norm
    # x_divide_std = config.x_divide_std
    # n_epochs = config.n_epochs
    # n_runs = config.n_runs
    # obs_data_path_list = config.obs_data_path_list
    # station_name_list = config.station_name_list

    dir_log_target = os.path.join(DIR_LOG, tag, target)
    make_dir(dir_log_target)

    if mode == 'run':
        data_generator_spatial = DataGeneratorSpatial(config.period, config.window, norm=config.norm,
                                                      x_divide_std=config.x_divide_std)
        for wid in range(TESTING_SLIDING_WINDOW, len(MONTH_LIST)):
            dir_log_exp = os.path.join(dir_log_target, str(MONTH_LIST[wid]))
            months = get_month_list(eval_mode, wid)
            data_generator_spatial.set_data(months)
            data_generator_spatial.prepare_data(config.target_size,
                                                train_step=config.train_step, test_step=config.test_step,
                                                single_step=config.single_step)
            batch_run(config.n_runs, dir_log_exp,
                      lambda dir_log_curr:
                      run_convlstm(config.station_name_list, dir_log_curr, data_generator_spatial, target,
                                   config.n_epochs, features_history, features_future, model_name))
    elif mode == 'reduce':
        csv_result_list = ['metrics_model_{}.csv'.format(model_name), 'metrics_nwp_{}.csv'.format(model_name)]
        reduce(csv_result_list, target, dir_log_target, config.n_runs, config.station_name_list)
