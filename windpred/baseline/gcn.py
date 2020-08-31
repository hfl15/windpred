import tensorflow as tf
import numpy as np
import os

from windpred.utils.base import DIR_LOG, make_dir
from windpred.utils.model_base import BasePredictor
from windpred.utils.data_parser import DataGeneratorV2Spatial
from windpred.utils.model_base import TESTING_SLIDING_WINDOW, MONTH_LIST, get_month_list, batch_run
from windpred.utils.model_base import reduce

from windpred.mhstn.base import get_data_spatial, run_spatial

from .gcn_spektral import GraphConv
from .gcn_spektral_utils import localpooling_filter, normalized_adjacency


class GCNLSTM(BasePredictor):
    def __init__(self, input_shape, units_output=24, verbose=1, name='gcn'):
        # input_shape = (seq_len, n_features, n_notes)
        super(GCNLSTM, self).__init__(input_shape, units_output, verbose, name)
        self.model = self.build_model()
        if self.verbose > 0:
            self.model.summary()

    def build_model(self):  # v8, refer to my local module, and spatial module
        seq_len, n_features, n_notes = self.input_shape[0], self.input_shape[1], self.input_shape[2]

        inp_seq = tf.keras.layers.Input((n_notes, seq_len))
        inp_feat = tf.keras.layers.Input((n_notes, n_features))
        inp_adj = tf.keras.layers.Input((n_notes, n_notes))

        x = GraphConv(64, activation='relu')([inp_feat, inp_adj])
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(x.shape[1] * 2, activation='relu')(x)

        xx = tf.keras.layers.LSTM(32, activation='relu')(inp_seq)
        xx = tf.keras.layers.Dense(xx.shape[1] * 2, activation='relu')(xx)

        x = tf.keras.layers.Concatenate()([x, xx])
        x = tf.keras.layers.Dense(x.shape[1], activation='relu')(x)
        out = tf.keras.layers.Dense(self.units_output)(x)

        model = tf.keras.models.Model([inp_seq, inp_feat, inp_adj], out)

        return model


def cal_corr(x, mode='corr'):
    n_notes = x.shape[1]
    if mode == 'constant':
        res = np.ones((x.shape[0], n_notes, n_notes))
    else:
        res = []
        for s in x:
            corr = np.corrcoef(s)
            res.append(corr)
        res = np.stack(res, axis=0)
    return res


def run(station_name_list, dir_log, data_generator_spatial, target, n_epochs,
        features_history=None, features_future=None, adjacency_norm='localpooling_filter', model_name='gcn'):
    tag_func = model_name
    n_stations = len(station_name_list)

    x_train_list, x_val_list, x_test_list, y_train_list, y_val_list, y_test_list = get_data_spatial(
        data_generator_spatial, station_name_list, target, features_history, features_future)

    def get_data(x_list):
        res = []
        for x in x_list:
            if type(x) == list and len(x) == 2:
                x_h, x_f = x
                x_ = np.hstack([x_h.reshape(x_h.shape[0], -1), x_f.reshape(x_f.shape[0], -1)])
            else:
                x_ = x
            res.append(x_.reshape((x_.shape[0], 1, -1)))
        res = np.concatenate(res, axis=1)
        return res  # (, n_notes, seq_length)
    x_tr = get_data(x_train_list)
    x_val = get_data(x_val_list)
    x_te = get_data(x_test_list)

    a_tr = cal_corr(x_tr)
    a_val = cal_corr(x_val)
    a_te = cal_corr(x_te)
    if adjacency_norm == 'localpooling_filter':
        a_tr = localpooling_filter(1 - np.abs(a_tr))
        a_val = localpooling_filter(1 - np.abs(a_val))
        a_te = localpooling_filter(1 - np.abs(a_te))
    elif adjacency_norm == 'normalized_adjacency':
        # https://github.com/danielegrattarola/spektral/blob/master/examples/graph_prediction/BDGC_disjoint.py
        a_tr_ = np.stack([normalized_adjacency(a) for a in a_tr])
        a_val_ = np.stack([normalized_adjacency(a) for a in a_val])
        a_te_ = np.stack([normalized_adjacency(a) for a in a_te])
        a_tr, a_val, a_te = a_tr_, a_val_, a_te_

    input_shape = (x_tr.shape[-1], x_tr.shape[-1], n_stations)
    x_tr = [x_tr, x_tr, a_tr]
    x_val = [x_val, x_val, a_val]
    x_te = [x_te, x_te, a_te]
    run_spatial(station_name_list, GCNLSTM, dir_log, data_generator_spatial, target, n_epochs,
                x_tr, x_val, x_te, y_train_list, y_val_list, y_test_list, input_shape, tag_func)


def main(target, mode, eval_mode, config, tag, model_name, features_history, features_future, adjacency_norm):
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
    make_dir(dir_log_target)

    if mode == 'run':
        data_generator_spatial = DataGeneratorV2Spatial(period, window, norm=norm, x_divide_std=x_divide_std)
        for wid in range(TESTING_SLIDING_WINDOW, len(MONTH_LIST)):
            dir_log_exp = os.path.join(dir_log_target, str(MONTH_LIST[wid]))
            months = get_month_list(eval_mode, wid)
            data_generator_spatial.set_data(months)
            data_generator_spatial.prepare_data(target_size,
                                                train_step=train_step, test_step=test_step, single_step=single_step)
            batch_run(n_runs, dir_log_exp,
                      lambda dir_log_curr: run(station_name_list, dir_log_curr, data_generator_spatial,
                                               target, n_epochs, features_history, features_future, adjacency_norm,
                                               model_name))
    elif mode == 'reduce':
        csv_result_list = ['metrics_model_{}.csv'.format(model_name), 'metrics_nwp_{}.csv'.format(model_name)]
        reduce(csv_result_list, target, dir_log_target, n_runs, station_name_list)

