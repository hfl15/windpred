import os
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV, LassoCV
from sklearn.preprocessing import minmax_scale
import json

from windpred.utils.base import tag_path, make_dir, DIR_LOG
from windpred.utils.data_parser import DataGenerator
from windpred.utils.model_base import DefaultConfig
from windpred.utils.exp import get_covariates_history_all


def calculate_weights(data_generator_list, dir_log, features, target, method='ridge'):
    def init_res():
        res = dict()
        for feat in features:
            res[feat] = []
        return res
    res = init_res()
    station_name_list = []
    for data_generator in data_generator_list:
        (x_train, y_train), (x_val, y_val), (_, _) = \
            data_generator.extract_training_data(x_attributes=features, y_attributes=[target])

        x_train = x_train.reshape((x_train.shape[0], -1))
        x_val = x_val.reshape((x_val.shape[0], -1))

        x_train = np.vstack([x_train, x_val])
        y_train = np.hstack([y_train, y_val])

        if method == 'ridge':
            model = RidgeCV(fit_intercept=False)
        elif method == 'lasso':
            model = LassoCV(fit_intercept=False)
        else:
            raise ValueError('the method={} can not be found!'.format(method))

        model.fit(x_train, y_train)
        print(model.coef_)
        for feat, coef in zip(features, model.coef_):
            res[feat].append(coef)

        station_name_list.append(data_generator.station_name)
    res = pd.DataFrame(res, index=station_name_list)
    res.to_csv(os.path.join(dir_log, '{}.csv'.format(method)))
    res_abs = res.abs()
    res_abs.to_csv(os.path.join(dir_log, '{}_abs.csv'.format(method)))

    def abs_min_max_scale(df):
        arr = df.values
        arr = np.abs(arr)
        arr = minmax_scale(arr.transpose()).transpose()
        res = pd.DataFrame(arr, columns=df.columns, index=df.index)
        return res
    res_min_max_scaled = abs_min_max_scale(res)
    res_min_max_scaled.to_csv(os.path.join(dir_log, '{}_abs_minmax.csv'.format(method)))

    return res_min_max_scaled


def select_features_fixed_threshold(dir_log, method='ridge'):
    df_weight = pd.read_csv(os.path.join(dir_log, '{}_weights_avg.csv'.format(method)), index_col=0)
    df_weight_agg = df_weight.mean()
    df_weight_agg.to_csv(os.path.join(dir_log, '{}_final_importance.csv'.format(method)))

    station_name_list = list(df_weight.index)
    threshold = 0.2
    features_selected = [feat for feat, weigh in zip(list(df_weight_agg.index), df_weight_agg.values)
                         if weigh > threshold]
    features_selected_dic = {}
    for station_name in station_name_list:
        features_selected_dic[station_name] = features_selected
    with open(os.path.join(dir_log, '{}_selected_features.json'.format(method)), 'w') as f:
        json.dump(features_selected_dic, f)


if __name__ == '__main__':
    tag = tag_path(os.path.abspath(__file__), 2)

    norm = DefaultConfig.norm
    x_divide_std = DefaultConfig.x_divide_std
    period = DefaultConfig.period
    window = 1
    train_step = 1
    test_step = 1
    single_step = True
    n_epochs = DefaultConfig.n_epochs
    obs_data_path_list = DefaultConfig.obs_data_path_list

    target = 'V'
    method = 'ridge'
    dir_log_root = os.path.join(DIR_LOG, tag, target)

    features = get_covariates_history_all()

    for target_size in range(1, period+1):
        dir_log = os.path.join(dir_log_root, str(target_size))
        make_dir(dir_log)
        data_generator_list = []
        for obs_data_path in obs_data_path_list:
            data_generator = DataGenerator(period, window, path=obs_data_path, norm=norm, x_divide_std=x_divide_std)
            data_generator.prepare_data(target_size,
                                        train_step=train_step, test_step=test_step, single_step=single_step)
            data_generator_list.append(data_generator)
        calculate_weights(data_generator_list, dir_log, features, target, method=method)

    res_min_max_scaled_sum = None
    for target_size in range(1, period+1):
        dir_log = os.path.join(dir_log_root, str(target_size))
        res_min_max_scaled = pd.read_csv(os.path.join(dir_log, '{}_abs_minmax.csv'.format(method)), index_col=0)
        if res_min_max_scaled_sum is None:
            res_min_max_scaled_sum = res_min_max_scaled
        else:
            res_min_max_scaled_sum = res_min_max_scaled_sum + res_min_max_scaled
    res_min_max_scaled_mean = res_min_max_scaled_sum/period
    res_min_max_scaled_mean.to_csv(os.path.join(dir_log_root, '{}_weights_avg.csv'.format(method)))

    select_features_fixed_threshold(dir_log_root, method)











