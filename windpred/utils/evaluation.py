import os
import pandas as pd
import numpy as np

from .base import make_dir
from . import evaluation_metrics


def dir_metrics(y_true, y_pred, with_delta=False):
    y_true = pd.DataFrame(y_true)
    y_pred = pd.DataFrame(y_pred)

    delta = y_pred - y_true
    delta[delta > 180] = delta - 360
    delta[delta < -180] = delta + 360

    bias = delta.mean()
    mae = np.abs(delta).mean()
    rmse = np.sqrt((delta ** 2.).mean())

    hit = (np.abs(delta) <= 20).astype(np.float)
    hit_rate = np.sum(hit) / len(hit)

    ret = [bias.values[0], rmse.values[0], mae.values[0], hit_rate.values[0]]

    if with_delta:
        ret.append(delta.values)

    return ret


def dir_relative_metrics(y_true, y_pred):
    bias, rmse, mae, hr, delta = dir_metrics(y_true, y_pred, with_delta=True)

    y_true = np.array(y_true)
    # divided by max. max of errors/deltas/bias = 180
    nmae_max = mae / 180
    nrmse_max = rmse / 180
    # divided by mean
    mean = np.mean(np.minimum(y_true, 360-y_true))
    nmae_mean = mae / mean
    nrmse_mean = rmse / mean

    return nmae_max, nrmse_max, nmae_mean, nrmse_mean


class Evaluator(object):
    metrics = ['big_rmse', 'small_rmse', 'all_rmse'] + ['mae', 'rae', 'rse', 'rrse']

    def __init__(self, dir_out, name):
        self.dir_out = os.path.join(dir_out, 'evaluate')
        make_dir(self.dir_out)
        self.name = name
        self.res = self.init_res()
        self.count = 0
        self.keys = []

    def init_res(self):
        res = dict()
        for m in self.metrics:
            res[m] = []
        return res

    def evaluate(self, y, y_pred, filter_big_wind):
        small_wind = [not b for b in filter_big_wind]

        rmse = evaluation_metrics.rmse(y, y_pred)
        big_rmse = evaluation_metrics.rmse(y[filter_big_wind], y_pred[filter_big_wind])
        small_rmse = evaluation_metrics.rmse(y[small_wind], y_pred[small_wind])

        print('{0} evaluation result:'.format(self.name))
        print("All wind:\trmse={0:.4f}".format(rmse))
        print("Big wind:\trmse={0:.4f}".format(big_rmse))
        print("Small wind:\trmse={0:.4f}".format(small_rmse))

        res = self.init_res()
        res['big_rmse'] = big_rmse
        res['small_rmse'] = small_rmse
        res['all_rmse'] = rmse

        # add more metrics
        res['mae'] = evaluation_metrics.mae(y, y_pred)
        res['rae'] = evaluation_metrics.rae(y, y_pred)
        res['rse'] = evaluation_metrics.rse(y, y_pred)
        res['rrse'] = evaluation_metrics.rrse(y, y_pred)

        return res

    def append(self, y, y_pred, filter_big_wind, key=None, flush=True):
        metrics = self.evaluate(y, y_pred, filter_big_wind)
        for k, v in metrics.items():
            self.res[k].append(v)

        if key is not None:
            self.keys.append(key)
        else:
            self.keys.append(self.count)
        self.count += 1

        if flush:
            self.save()

        return metrics

    def get_df(self):
        return pd.DataFrame(self.res, index=self.keys)

    def save(self):
        df = self.get_df()
        df.to_csv(os.path.join(self.dir_out, 'metrics_{}.csv'.format(self.name)))
        return df


class EvaluatorDir(Evaluator):
    metrics = ['big_bias', 'big_rmse', 'big_mae', 'big_hit',
               'small_bias', 'small_rmse', 'small_mae', 'small_hit',
               'all_bias', 'all_rmse', 'all_mae', 'all_hit'] \
              + ['nmae_max', 'nrmse_max', 'nmae_mean', 'nrmse_mean']

    def __init__(self, dir_out, name):
        super(EvaluatorDir, self).__init__(dir_out, name)
        self.res = self.init_res()

    def evaluate(self, y, y_pred, filter_big_wind):
        small_wind = [not b for b in filter_big_wind]

        bias, rmse, mae, hit = dir_metrics(y, y_pred)
        big_bias, big_rmse, big_mae, big_hit = dir_metrics(y[filter_big_wind], y_pred[filter_big_wind])
        small_bias, small_rmse, small_mae, small_hit = dir_metrics(y[small_wind], y_pred[small_wind])

        print('{0} evaluation result:'.format(self.name))
        print("All wind:\tbias={0:.4f},\trmse={1:.4f},\tmae={2:.4f},\thit={3:.4f}".format(bias, rmse, mae, hit))
        print("Big wind:\tbias={0:.4f},\trmse={1:.4f},\tmae={2:.4f},\thit={3:.4f}".format(big_bias, big_rmse, big_mae, big_hit))
        print("Small wind:\tbias={0:.4f},\trmse={1:.4f},\tmae={2:.4f},\thit={3:.4f}".format(small_bias, small_rmse, small_mae, small_hit))

        res = self.init_res()
        res['big_bias'] = big_bias
        res['big_rmse'] = big_rmse
        res['big_mae'] = big_mae
        res['big_hit'] = big_hit

        res['small_bias'] = small_bias
        res['small_rmse'] = small_rmse
        res['small_mae'] = small_mae
        res['small_hit'] = small_hit

        res['all_bias'] = bias
        res['all_rmse'] = rmse
        res['all_mae'] = mae
        res['all_hit'] = hit

        # add more metrics
        nmae_max, nrmse_max, nmae_mean, nrmse_mean = dir_relative_metrics(y, y_pred)
        res['nmae_max'] = nmae_max
        res['nrmse_max'] = nrmse_max
        res['nmae_mean'] = nmae_mean
        res['nrmse_mean'] = nrmse_mean

        return res
