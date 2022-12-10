import os
import numpy as np
import keras.backend as K

from windpred.utils.data_parser import DataGeneratorSpatial
from windpred.utils.model_base import DefaultConfig
from windpred.utils.evaluation import Evaluator
from windpred.mhstn.base import get_evaluation_data_spatial

from windpred.baseline.duq.config import ParameterConfig
from windpred.baseline.duq.seq2seq_class import Seq2Seq_Class
from keras.models import model_from_json


def get_data(data_generator, station_name_list, target, features_history_in, features_future_in):
    xh_train_all, xh_val_all, xh_test_all = [], [], []
    xf_train_all, xf_val_all, xf_test_all = [], [], []
    y_train_all, y_val_all, y_test_all = [], [], []
    for station_idx, station_name in enumerate(station_name_list):

        if type(features_history_in) == dict:
            features_history = features_history_in[station_name]
        else:
            features_history = features_history_in
        if type(features_future_in) == dict:
            features_future = features_future_in[station_name]
        else:
            features_future = features_future_in

        y_attributes = ['{}_S{}'.format(target, station_idx)]
        x_attributes = ['{}_S{}'.format(f, station_idx) for f in features_history] + features_future
        (x_train, y_train), (x_val, y_val), (x_test, y_test) = \
            data_generator.extract_training_data(x_attributes=x_attributes, y_attributes=y_attributes)

        xh_train, xf_train = x_train[:, :, :len(features_history)], x_train[:, :, len(features_history):]
        xh_val, xf_val = x_val[:, :, :len(features_history)], x_val[:, :, len(features_history):]
        xh_test, xf_test = x_test[:, :, :len(features_history)], x_test[:, :, len(features_history):]

        xh_train_all.append(xh_train)
        xf_train_all.append(xf_train)
        y_train_all.append(np.expand_dims(y_train, -1))
        xh_val_all.append(xh_val)
        xf_val_all.append(xf_val)
        y_val_all.append(np.expand_dims(y_val, -1))
        xh_test_all.append(xh_test)
        xf_test_all.append(xf_test)
        y_test_all.append(np.expand_dims(y_test, -1))

    xh_train_all = np.stack(xh_train_all, axis=-2)
    xf_train_all = np.stack(xf_train_all, axis=-2)
    y_train_all = np.stack(y_train_all, axis=-2)
    xh_val_all = np.stack(xh_val_all, axis=-2)
    xf_val_all = np.stack(xf_val_all, axis=-2)
    y_val_all = np.stack(y_val_all, axis=-2)
    xh_test_all = np.stack(xh_test_all, axis=-2)
    xf_test_all = np.stack(xf_test_all, axis=-2)
    y_test_all = np.stack(y_test_all, axis=-2)

    return (xh_train_all, xf_train_all, y_train_all), (xh_val_all, xf_val_all, y_val_all), \
           (xh_test_all, xf_test_all, y_test_all)


class DUQPredictor(object):
    """
        To leave the original configuration as unchanged as possible,
        this class obeys the DUQ primitive named system.
    """
    def __init__(self,
                 target: str,
                 dir_log: str,
                 param: ParameterConfig,
                 data_generator: DataGeneratorSpatial,
                 station_name_list: list):
        self.target = target
        self.dir_log = dir_log
        self.param = param
        self.data_generator = data_generator
        self.station_name_list = station_name_list

    def train(self, train_dict, val_dict, model_name='Seq2Seq_MVE'):

        enc_dec = Seq2Seq_Class(self.param, self.dir_log, model_structure_name=model_name,
                                model_weights_name=model_name, model_name=model_name)
        enc_dec.build_graph()

        n_stations = train_dict['input_obs'].shape[-2]
        n_future_horizons = train_dict['ground_truth'].shape[1]
        n_samples_val = val_dict['input_ruitu'].shape[0]
        val_ids = []
        val_times = []
        for i in range(n_stations):
            val_ids.append(np.ones(shape=(n_samples_val, n_future_horizons)) * i)
        val_ids = np.stack(val_ids, axis=-1)
        print('val_ids.shape is:', val_ids.shape)
        val_times = np.array(range(n_future_horizons))
        val_times = np.tile(val_times, (n_samples_val, 1))
        print('val_times.shape is:', val_times.shape)

        enc_dec.fit(train_dict['input_obs'], train_dict['input_ruitu'], train_dict['ground_truth'],
                    val_dict['input_obs'], val_dict['input_ruitu'], val_dict['ground_truth'],
                    val_ids=val_ids, val_times=val_times, iterations=10000, batch_size=512, validation=True)

        print('Training finished!')

    def _predict(self, model, batch_inputs, batch_ruitu, batch_ids, batch_times):
        pred_result_list = []
        pred_var_list = []
        for i in range(self.param.num_stations):
            result = model.predict(
                x=[batch_inputs[:, :, i, :], batch_ruitu[:, :, i, :], batch_ids[:, :, i], batch_times])
            var_result = result[:, :, self.param.num_variables_to_predict:self.param.num_statistics_to_predict]  # Variance
            result = result[:, :, :self.param.num_variables_to_predict]  # Mean
            pred_result_list.append(result)
            pred_var_list.append(var_result)

        pred_result = np.stack(pred_result_list, axis=2)
        pred_var_result = np.stack(pred_var_list, axis=2)
        pred_std = np.sqrt(pred_var_result)
        return pred_result, pred_std

    def predict(self, test_dict, model_name='Seq2Seq_MVE_layers_50_50_loss_mae_dropout0'):
        saved_csv_name = model_name + '.csv'

        # load json and create model
        json_file = open(os.path.join(self.dir_log, '{}.json'.format(model_name)), 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        model = model_from_json(loaded_model_json)
        print(model.summary())
        # load weights into new model
        model.load_weights(os.path.join(self.dir_log, '{}.h5'.format(model_name)))

        test_inputs = test_dict['input_obs']
        test_ruitu = test_dict['input_ruitu']
        n_samples = test_inputs.shape[0]

        # add station ids
        test_ids = []
        for i in range(self.param.num_stations):
            test_ids.append(np.ones(self.param.num_steps_to_predict) * i)
        test_ids = np.stack(test_ids, axis=-1)
        test_ids = np.broadcast_to(test_ids, (n_samples,)+test_ids.shape)

        # add time
        test_times = np.array(range(self.param.num_steps_to_predict))
        test_times = np.tile(test_times, (n_samples, 1))

        pred_result, pred_std_result = self._predict(model, test_inputs, test_ruitu, test_ids, test_times)

        return pred_result, pred_std_result

    def test(self, test_dict, model_name='Seq2Seq_MVE_layers_50_50_loss_mae_dropout0'):
        pred_result, pred_std_result = self.predict(test_dict, model_name)

        nwp, obs_list, speed_list, filter_big_wind_list = get_evaluation_data_spatial(
            self.data_generator, self.target, self.param.num_stations)
        evaluator_model = Evaluator(self.dir_log, 'model')
        evaluator_nwp = Evaluator(self.dir_log, 'nwp')
        for i_station in range(self.param.num_stations):
            station_name = self.station_name_list[i_station]
            y_pred = pred_result[:, :, i_station, :].ravel()

            if self.data_generator.norm is not None:
                target_curr = '{}_S{}'.format(self.target, i_station)
                y_pred = self.data_generator.normalizer.inverse_transform(target_curr, y_pred)

            obs = obs_list[i_station]
            filter_big_wind = filter_big_wind_list[i_station]
            evaluator_model.append(obs, y_pred, filter_big_wind, key=station_name)
            evaluator_nwp.append(obs, nwp, filter_big_wind, key=station_name)

            np.savetxt(os.path.join(self.dir_log, "y_pred_test_{}.txt".format(station_name)), y_pred)


def run(features_history, features_future, loss='mae', layers=[50, 50]):
    def _run(config: DefaultConfig, dir_log, data_generator_spatial, target):
        train_data, val_data, test_data = get_data(data_generator_spatial, config.station_name_list, target,
                                                   features_history, features_future)

        train_dict = {'input_obs': train_data[0], 'input_ruitu': train_data[1], 'ground_truth': train_data[2]}
        val_dict = {'input_obs': val_data[0], 'input_ruitu': val_data[1], 'ground_truth': val_data[2]}
        test_dict = {'input_obs': test_data[0], 'input_ruitu': test_data[1], 'ground_truth': test_data[2]}

        param = ParameterConfig()
        param.num_input_features = len(features_history)
        param.num_output_features = 1
        param.num_decoder_features = len(features_future)
        param.input_sequence_length = config.window
        param.target_sequence_length = config.target_size
        param.num_steps_to_predict = config.target_size
        param.num_stations = len(config.station_name_list)
        param.num_variables_to_predict = 1
        param.num_statistics_to_predict = param.num_variables_to_predict * 2
        # Critical hyper-parameters.
        # Note: These settings will not change the initialization model name. Therefore, the default names used in DUQ
        # doesn't need to be changed.
        param.loss = loss
        param.layers = layers

        duq = DUQPredictor(target, dir_log, param, data_generator_spatial, config.station_name_list)
        duq.train(train_dict, val_dict)
        duq.test(test_dict)
        K.clear_session()

    return _run


