import numpy as np
import tensorflow as tf

from windpred.utils.model_base import BaseLSTM
from windpred.mhstn.base import get_data_spatial, run_spatial


class LSTM2Layers(BaseLSTM):
    def __init__(self, input_shape, units_output=24, verbose=1, name='lstm2'):
        super(BaseLSTM, self).__init__(input_shape, units_output, verbose, name)

        self.model = self.build_model(self.input_shape, self.units_output)
        if self.verbose > 0:
            self.model.summary()

    @staticmethod
    def build_model(input_shape, units_output, units_lstm=32):
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.LSTM(units_lstm, input_shape=input_shape, return_sequences=True))
        model.add(tf.keras.layers.LSTM(units_lstm))
        model.add(tf.keras.layers.Dense(units_output))
        return model


def run_lstm(features_history=None, features_future=None, concat_mode='cascade', model_mode='lstm1'):
    def _run_lstm(station_name_list, dir_log, data_generator_spatial, target, n_epochs):

        x_train_list, x_val_list, x_test_list, y_train_list, y_val_list, y_test_list = get_data_spatial(
            data_generator_spatial, station_name_list, target, features_history, features_future)

        def get_data(x_list):
            res = []
            for x in x_list:
                if type(x) == list and len(x) == 2:
                    x_h, x_f = x
                    if concat_mode == 'parallel':
                        x_ = np.concatenate([x_h, x_f], axis=-1)
                    elif concat_mode == 'cascade':
                        x_ = np.concatenate([x_h, x_f], axis=1)
                    else:
                        raise ValueError("concat_mode = {} can not be found!".format(concat_mode))
                else:
                    x_ = x
                res.append(x_)
            res = np.stack(res, axis=-1)
            res = res.reshape((res.shape[0], res.shape[1], -1))  # (#samples, seq_len, n_features*n_channels)
            return res

        x_tr = get_data(x_train_list)
        x_val = get_data(x_val_list)
        x_te = get_data(x_test_list)
        input_shape = x_tr.shape[1:]

        if model_mode == 'lstm1':
            cls_model = BaseLSTM
        elif model_mode == 'lstm2':
            cls_model = LSTM2Layers
        else:
            raise ValueError('model_mode = {} can not be found!'.format(model_mode))

        run_spatial(station_name_list, cls_model, dir_log, data_generator_spatial, target, n_epochs,
                    x_tr, x_val, x_te, y_train_list, y_val_list, y_test_list, input_shape)
    return _run_lstm
