import os
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error

from .base import make_dir


def cal_dir_metrics(obs, pred):
    obs = pd.DataFrame(obs)
    pred = pd.DataFrame(pred)
    delta = pred - obs
    delta[delta > 180] = delta - 360
    delta[delta < -180] = delta + 360
    bias = delta.mean()
    mae = np.abs(delta).mean()
    rmse = np.sqrt((delta ** 2.).mean())

    hit = (np.abs(delta) <= 20).astype(np.float)
    hit_rate = np.sum(hit) / len(hit)

    return bias.values[0], rmse.values[0], mae.values[0], hit_rate.values[0]


def cal_rmse(obs, pred):
    return np.sqrt(mean_squared_error(obs, pred))


def cal_delta(obs, pred):
    obs = pd.DataFrame(obs)
    pred = pd.DataFrame(pred)
    delta = pred - obs
    delta[delta > 180] = delta - 360
    delta[delta < -180] = delta + 360

    return delta.values


def cal_hitrate(obs, pred):
    delta_abs = np.abs(cal_delta(obs, pred))
    hit = (delta_abs<=20).astype(np.float)
    hit_rate = np.sum(hit) / len(hit)
    return hit_rate


def evaluate_mae(desc, y, y_pred, big_wind):
    small_wind = [not i for i in big_wind]

    bias, rmse, mae = cal_dir_metrics(y, y_pred)
    big_wind_bias, big_wind_rmse, big_wind_mae = cal_dir_metrics(y[big_wind], y_pred[big_wind])
    small_wind_bias, small_wind_rmse, small_wind_mae = cal_dir_metrics(y[small_wind], y_pred[small_wind])

    print('{0} evaluation result:'.format(desc))
    print("All wind:\tbias={0:.4f},\trmse={1:.4f},\tmae={2:.4f}".format(bias, rmse, mae))
    print("Big wind:\tbias={0:.4f},\trmse={1:.4f},\tmae={2:.4f}".format(big_wind_bias, big_wind_rmse,
                                                                          big_wind_mae))
    print("Small wind:\tbias={0:.4f},\trmse={1:.4f},\tmae={2:.4f}".format(small_wind_bias, small_wind_rmse,
                                                                            small_wind_mae))

    if 'NWP' in desc:
        with open('nwp_big_wind.txt', 'a') as file:
            file.write('{0:.4f} | '.format(big_wind_mae))
        with open('nwp_small_wind.txt', 'a') as file:
            file.write('{0:.4f} | '.format(small_wind_mae))
        with open('nwp_mae.txt', 'a') as file:
            file.write('{0:.4f} | '.format(mae))
    else:
        with open('big_wind.txt', 'a') as file:
            file.write('{0:.4f} | '.format(big_wind_mae))
        with open('small_wind.txt', 'a') as file:
            file.write('{0:.4f} | '.format(small_wind_mae))
        with open('mae.txt', 'a') as file:
            file.write('{0:.4f} | '.format(mae))

    return big_wind_rmse


def evaluate_rmse(desc, y, y_pred, big_wind):
    small_wind = [not i for i in big_wind]

    rmse = cal_rmse(y, y_pred)
    big_wind_rmse = cal_rmse(y[big_wind], y_pred[big_wind])
    small_wind_rmse = cal_rmse(y[small_wind], y_pred[small_wind])

    print('{0} evaluation result:'.format(desc))
    print("All wind:\trmse={0:.4f}".format(rmse))
    print("Big wind:\trmse={0:.4f}".format(big_wind_rmse))
    print("Small wind:\trmse={0:.4f}".format(small_wind_rmse))

    return big_wind_rmse


class Evaluator(object):
    metrics = ['big_rmse', 'small_rmse', 'all_rmse']

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

        rmse = cal_rmse(y, y_pred)
        big_rmse = cal_rmse(y[filter_big_wind], y_pred[filter_big_wind])
        small_rmse = cal_rmse(y[small_wind], y_pred[small_wind])

        print('{0} evaluation result:'.format(self.name))
        print("All wind:\trmse={0:.4f}".format(rmse))
        print("Big wind:\trmse={0:.4f}".format(big_rmse))
        print("Small wind:\trmse={0:.4f}".format(small_rmse))

        res = self.init_res()
        res['big_rmse'] = big_rmse
        res['small_rmse'] = small_rmse
        res['all_rmse'] = rmse

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
               'all_bias', 'all_rmse', 'all_mae', 'all_hit']

    def __init__(self, dir_out, name):
        super(EvaluatorDir, self).__init__(dir_out, name)
        self.res = self.init_res()

    def evaluate(self, y, y_pred, filter_big_wind):
        small_wind = [not b for b in filter_big_wind]

        bias, rmse, mae, hit = cal_dir_metrics(y, y_pred)
        big_bias, big_rmse, big_mae, big_hit = cal_dir_metrics(y[filter_big_wind], y_pred[filter_big_wind])
        small_bias, small_rmse, small_mae, small_hit = cal_dir_metrics(y[small_wind], y_pred[small_wind])

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

        return res
