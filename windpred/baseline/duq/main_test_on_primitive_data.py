"""
    test duq on the data used in the primitive paper.
"""
import pandas as pd
import os
import numpy as np

from windpred.utils.base import tag_path, make_dir, DIR_LOG

from windpred.baseline.duq.helper import load_pkl
from windpred.baseline.duq.main import DUQPredictor
from windpred.baseline.duq.config import ParameterConfig


def get_training_data():
    processed_path = '../../../cache/Deep_Learning_Weather_Forecasting/data/processed/'
    train_data = 'train_norm.dict'
    val_data = 'val_norm.dict'

    # [Fanling]
    # keys : ['input_obs', 'input_ruitu', 'ground_truth']
    # train_dict['input_obs'].shape : (1148, 37, 10, 9)
    # train_dict['input_ruitu'].shape : (1148, 37, 10, 29)
    # train_dict['ground_truth'].shape : (1148, 37, 10, 3)
    train_dict = load_pkl(processed_path, train_data)
    # [Fanling]
    # keys : ['input_obs', 'input_ruitu', 'ground_truth']
    # val_dict['input_obs'].shape : (87, 37, 10, 9)
    # val_dict['input_ruitu'].shape : (87, 37, 10, 29)
    # val_dict['ground_truth'].shape : (87, 37, 10, 3)
    val_dict = load_pkl(processed_path, val_data)

    print(train_dict.keys())
    print('Original input_obs data shape:')
    print(train_dict['input_obs'].shape)
    print(val_dict['input_obs'].shape)
    print(train_dict['ground_truth'].shape)

    print('After clipping the 9 days, input_obs data shape:')
    train_dict['input_obs'] = train_dict['input_obs'][:, :-9, :, :]  # [Fanling] shape=(1148, 28, 10, 9)
    val_dict['input_obs'] = val_dict['input_obs'][:, :-9, :, :]  # [Fanling] shape=(87, 28, 10, 9)
    print(train_dict['input_obs'].shape)
    print(val_dict['input_obs'].shape)
    print(val_dict['ground_truth'].shape)

    return train_dict, val_dict


def get_testing_data():
    processed_path = '../../../cache/Deep_Learning_Weather_Forecasting/data/processed/'
    test_file_name = 'OnlineEveryDay_20181028_norm.dict'

    test_file = test_file_name
    # [fanling]
    # keys = ['input_obs', 'input_ruitu']
    # test_data['input_obs'].shape = (28, 10, 9)
    # test_data['input_ruitu'].shape = (37, 10, 29)
    test_data = load_pkl(processed_path, test_file)

    test_data['input_obs'] = np.expand_dims(test_data['input_obs'], axis=0)
    test_data['input_ruitu'] = np.expand_dims(test_data['input_ruitu'], axis=0)

    return test_data


def save_result_to_csv(pred_result, dir_log, fname, param):
    res = {}
    for i_v in range(param.num_variables_to_predict):
        for i_s in range(param.num_stations):
            pred = pred_result[0, :, i_s, i_v]
            res['{}_{}'.format(i_v, i_s)] = pred
    res = pd.DataFrame(res)
    res.to_csv(os.path.join(dir_log, '{}.csv'.format(fname)), index=False)


if __name__ == '__main__':
    tag = tag_path(os.path.abspath(__file__), 2)

    dir_log = make_dir(os.path.join(DIR_LOG, tag))

    # set customized parameters
    param = ParameterConfig()
    param.num_input_features = 9
    param.num_output_features = 3
    param.num_decoder_features = 29
    param.input_sequence_length = 28
    param.target_sequence_length = 37
    param.num_steps_to_predict = 37
    param.num_stations = 10
    param.num_variables_to_predict = 3
    param.num_statistics_to_predict = 6

    # initialization
    duq = DUQPredictor('duq', dir_log, param, None, None)

    # train model
    train_dict, val_dict = get_training_data()
    duq.train(train_dict, val_dict)

    # test model
    test_dict = get_testing_data()
    pred_result, pred_std_result = duq.predict(test_dict)
    save_result_to_csv(pred_result, dir_log, 'pred_result', param)
    save_result_to_csv(pred_std_result, dir_log, 'pred_std_result', param)

    # compare our results with the results derived by the primitive program.
    primitive_model = 'Seq2Seq_MVE_layers_50_50_loss_mae_dropout0-2018102803_demo'
    tag_file = 'pred_result'
    df_pred_result = pd.read_csv(os.path.join(dir_log, '{}.csv'.format(tag_file)))
    df_pred_result_primitive = pd.read_csv(os.path.join(dir_log, '{}_{}.csv'.format(primitive_model, tag_file)))
    err = abs(df_pred_result - df_pred_result_primitive)
    print("mean error on {}.csv: ".format(tag_file), err.mean().mean())
    # mean error on pred_result.csv:  0.0002041767585585587
    tag_file = 'pred_std_result'
    df_pred_std_result = pd.read_csv(os.path.join(dir_log, 'pred_result.csv'))
    df_pred_std_result_primitive = pd.read_csv(os.path.join(dir_log, '{}_{}.csv'.format(primitive_model, tag_file)))
    err = abs(df_pred_std_result - df_pred_std_result_primitive)
    print("mean error on {}.csv: ".format(tag_file), err.mean().mean())
    # mean error on pred_std_result.csv:  0.5185171448261262



