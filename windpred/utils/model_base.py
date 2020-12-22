from sklearn.base import BaseEstimator
import os
import pandas as pd
import numpy as np
from time import time

from .base import get_station_name, get_files_path, get_tf_keras, make_dir
from .base import plot_train_valid_loss, plot_and_save_comparison
from .evaluation import Evaluator


tf, K = get_tf_keras()

"""
    default settings
"""


class DefaultConfig(object):
    target_size = 24
    period = 24
    window = period
    train_step = 24
    test_step = 24
    single_step = False

    norm = 'submean'
    x_divide_std = True

    n_epochs = 1000
    n_runs = 2
    # n_runs = 10

    obs_data_path_list, nwp_path = get_files_path()
    station_name_list = [get_station_name(path) for path in obs_data_path_list]


"""
    base models
"""


class BasePredictor(BaseEstimator):
    def __init__(self, input_shape, units_output=24, verbose=1, name='base'):
        self.input_shape = input_shape
        self.units_output = units_output
        self.verbose = verbose
        self.name = name
        self.model = None

    def fit(self, x, y,
            batch_size=32,
            n_epochs=20,
            validation_data=None,
            set_cb=True):
        callbacks = None
        if set_cb:
            reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
                factor=0.5, patience=3, min_lr=0.0001, mode='min', verbose=self.verbose)
            es = tf.keras.callbacks.EarlyStopping(patience=30, mode='min', verbose=self.verbose,
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

    def predict(self, x):
        return self.model.predict(x)

    def plot_train_history(self, dir_log, title='Training_Validation_Loss'):
        loss = self.history.history['loss']
        val_loss = self.history.history['val_loss']
        plot_train_valid_loss(loss, val_loss, dir_log, title)

    def save(self, path, name=None):
        if name is not None:
            dir_saved = os.path.join(path, '{}.hdf5'.format(name))
        else:
            dir_saved = os.path.join(path, '{}.hdf5'.format(self.name))
        self.model.save(dir_saved)

    def load(self, path, name=None):
        if name is not None:
            dir_saved = os.path.join(path, '{}.hdf5'.format(name))
        else:
            dir_saved = os.path.join(path, '{}.hdf5'.format(self.name))
        self.model = tf.keras.models.load_model(dir_saved)


class BaseLSTM(BasePredictor):
    def __init__(self, input_shape, units_output=24, verbose=1, name='base_lstm'):
        super(BaseLSTM, self).__init__(input_shape, units_output, verbose, name)

        self.model = self.build_model(self.input_shape, self.units_output)
        if self.verbose > 0:
            self.model.summary()

    @staticmethod
    def build_model(input_shape, units_output, units_lstm=32):
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.LSTM(units_lstm, input_shape=input_shape))
        model.add(tf.keras.layers.Dense(units_output))
        return model


class BaseMLP(BasePredictor):
    def __init__(self, input_shape, units_output=24, verbose=1, name='base_lstm'):
        super(BaseMLP, self).__init__(input_shape, units_output, verbose, name)

        self.model = self.build_model(self.input_shape, self.units_output)
        if self.verbose > 0:
            self.model.summary()

    @staticmethod
    def build_model(input_shape, units_output):
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Flatten(input_shape=input_shape))
        model.add(tf.keras.layers.Dense(input_shape[0] * 2, activation='relu'))
        model.add(tf.keras.layers.Dense(units_output))
        return model


class CombinerDense(BasePredictor):
    def __init__(self, input_shape, units_output=24, verbose=1, name='combiner_dense'):
        super(CombinerDense, self).__init__(input_shape, units_output=units_output, verbose=verbose, name=name)

        self.model = self.build_model(self.input_shape, self.units_output)
        if self.verbose > 0:
            self.model.summary()

    def build_model(self, input_shape, units_output):
        model = tf.keras.models.Sequential()
        model.add(
            tf.keras.layers.Dense(
                units_output, use_bias=False,
                kernel_initializer=tf.keras.initializers.Constant(1.0/input_shape[0]),
                input_shape=input_shape))
        return model


"""
    for an experiment
"""


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
        np.savetxt(os.path.join(dir_log, 'y_pred_train_{}.txt'.format(station_name+file_suffix)), model.predict(x_train))
        np.savetxt(os.path.join(dir_log, 'y_pred_val_{}.txt'.format(station_name+file_suffix)), model.predict(x_val))
        np.savetxt(os.path.join(dir_log, 'y_pred_test_{}.txt'.format(station_name+file_suffix)), model.predict(x_test))

    return time_training, time_inference


def reduce(csv_result_list, target, dir_log_target, n_runs, station_name_list):
    for wid in range(TESTING_SLIDING_WINDOW, len(MONTH_LIST)):
        dir_log_exp = os.path.join(dir_log_target, str(MONTH_LIST[wid]))
        if target == 'DIR':
            reduce_multiple_runs_dir(dir_log_exp, csv_result_list, n_runs, station_name_list)
        else:
            reduce_multiple_runs(dir_log_exp, csv_result_list, n_runs, station_name_list)

    csv_list = ['{}_agg_mean.csv'.format(f.split('.')[0]) for f in csv_result_list]
    if target == 'DIR':
        reduce_multiple_splits_dir(dir_log_target, csv_list)
    else:
        reduce_multiple_splits(dir_log_target, csv_list)


"""
    for experiments: multiple runs 
"""


def batch_run(n_runs, dir_log, func):
    for r in range(n_runs):
        K.clear_session()
        dir_log_curr = os.path.join(dir_log, str(r))
        make_dir(dir_log_curr)
        func(dir_log_curr)
    

def reduce_multiple_runs(dir_log, csv_list, n_runs, station_name_list, columns=['all_rmse', 'big_rmse', 'small_rmse']):
    for csv in csv_list:
        df_list = []
        for r in range(n_runs):
            df = pd.read_csv(os.path.join(dir_log, str(r), 'evaluate', csv), index_col=0)
            df_list.append(df)
        df_mean = pd.DataFrame(columns=columns)
        df_std = pd.DataFrame(columns=columns)
        for station_name in station_name_list:
            df_reduced = {}
            for col in columns:
                df_reduced[col] = []
            for r in range(n_runs):
                for col in columns:
                    df_reduced[col].append(df_list[r].loc[station_name, col])
            df_reduced = pd.DataFrame(df_reduced)
            mean = pd.DataFrame(df_reduced.mean().to_dict(), index=[station_name])
            std = pd.DataFrame(df_reduced.std().to_dict(), index=[station_name])
            df_reduced = df_reduced.append(mean, ignore_index=True)
            df_mean = df_mean.append(mean)
            df_std = df_std.append(std)
            df_reduced.to_csv(os.path.join(dir_log, '{}_{}.csv'.format(csv.split('.')[0], station_name)), index=False)
        df_mean.to_csv(os.path.join(dir_log, '{}_agg_mean.csv'.format(csv.split('.')[0])))
        df_std.to_csv(os.path.join(dir_log, '{}_agg_std.csv'.format(csv.split('.')[0])))

    print("Finish to reduce the results below the directory {}".format(dir_log))


def reduce_multiple_runs_dir(dir_log, csv_list, n_runs, station_name_list):
    columns = {'all_mae', 'big_mae', 'small_mae'}
    reduce_multiple_runs(dir_log, csv_list, n_runs, station_name_list, columns)


"""
    for experiments: multiple splits of the dataset 
"""

MONTH_LIST = [201803, 201804, 201805, 201806, 201807, 201808, 201809, 201810, 201811, 201812,
              201901, 201902, 201903, 201904, 201905, 201906, 201907, 201908]  # hard code for test
TESTING_SLIDING_WINDOW = 6


def get_month_list(eval_mode, wid):
    if eval_mode == 'rolling':
        months = MONTH_LIST[(wid - TESTING_SLIDING_WINDOW):(wid + 1)]
    elif eval_mode == 'increment':
        months = MONTH_LIST[:(wid + 1)]
    else:
        raise ValueError("eval_mode={} can not be found!".format(eval_mode))
    return months


def reduce_multiple_splits(path, csvf_list, col_name='all_rmse'):
    """
    Example:
        path = "cache/aaai21/aaai21_mhstn_covar_self/V"
        csvf_list = ['{}_agg_mean.csv'.format(f.split('.')[0]) for f in CSV_LIST]
    :param path:
    :param csvf_list:
    :param col_name:
    :return:
    """
    for csvf in csvf_list:
        dir_list = sorted([d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))])
        df_list = []
        for month in dir_list:
            df = pd.read_csv(os.path.join(path, month, csvf), index_col=0)
            df = df[col_name]
            df_list.append(df)
        dfc = pd.concat(df_list, axis=1, keys=dir_list)
        dfc.to_csv(os.path.join(path, csvf))
        print("Finish to processing {}".format(csvf))


def reduce_multiple_splits_dir(path, csvf_list, col_name='all_mae'):
    reduce_multiple_splits(path, csvf_list, col_name)



