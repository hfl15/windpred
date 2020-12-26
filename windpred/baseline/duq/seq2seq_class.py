import numpy as np
import keras
import tensorflow as tf
from matplotlib import pyplot as plt
import pandas as pd
from keras.layers.embeddings import Embedding
from keras.layers import concatenate, Lambda
import os, sys
from .weather_model import Seq2Seq_MVE_subnets_swish, Seq2Seq_MVE, Seq2Seq_MVE_subnets
from keras.models import load_model, model_from_json
from .config import ParameterConfig


def crop(dimension, start, end):
    # Crops (or slices) a Tensor on a given dimension from start to end
    # example : to crop tensor x[:, :, 5:10]
    # call slice(2, 5, 10) as you want to crop on the second dimension
    def func(x):
        if dimension == 0:
            return x[start: end]
        if dimension == 1:
            return x[:, start: end]
        if dimension == 2:
            return x[:, :, start: end]
        if dimension == 3:
            return x[:, :, :, start: end]
        if dimension == 4:
            return x[:, :, :, :, start: end]

    return Lambda(func)


class Seq2Seq_Class(object):
    def __init__(self,
                 param: ParameterConfig,
                 model_save_path: str,
                 model_structure_name='seq2seq_model_demo',
                 model_weights_name='seq2seq_model_demo',
                 model_name=None,
                 save_model=False):
        self.param = param
        self.model_save_path = model_save_path
        self.model_structure_name = model_structure_name + self.param.model_name_format_str + '.json'
        self.model_weights_name = model_weights_name + self.param.model_name_format_str + '.h5'
        print('model_structure_name:', self.model_structure_name)
        print('model_weights_name:', self.model_weights_name)

        self.pred_result = None  # Predicted mean value
        self.pred_var_result = None  # Predicted variance value
        self.current_mean_val_loss = None
        self.EARLY_STOP = False
        self.val_loss_list = []
        self.train_loss_list = []
        self.pred_var_result = []

    def build_graph(self):
        # keras.backend.clear_session() # clear session/graph
        self.optimizer = keras.optimizers.Adam(lr=self.param.lr, decay=self.param.decay)

        self.model = Seq2Seq_MVE(id_embd=self.param.id_embd, time_embd=self.param.time_embd,
                                 lr=self.param.lr, decay=self.param.decay,
                                 num_input_features=self.param.num_input_features,
                                 num_output_features=self.param.num_output_features,
                                 num_decoder_features=self.param.num_decoder_features,
                                 num_stations=self.param.num_stations,
                                 num_steps_to_predict=self.param.num_steps_to_predict,
                                 layers=self.param.layers,
                                 loss=self.param.loss, regulariser=self.param.regulariser,
                                 dropout_rate=self.param.dropout_rate)

        def loss_fn(y_true, y_pred):
            n_dim = len(y_pred.shape)
            id_dim_last = n_dim - 1

            pred_u = crop(id_dim_last, 0, self.param.num_variables_to_predict)(y_pred)  # mean of Gaussian distribution
            pred_sig = crop(id_dim_last, self.param.num_variables_to_predict, self.param.num_statistics_to_predict)(y_pred)  # variance of Gaussian distribution
            if self.param.loss == 'mve':
                precision = 1. / pred_sig
                log_loss = 0.5 * tf.log(pred_sig) + 0.5 * precision * ((pred_u - y_true) ** 2)
                log_loss = tf.reduce_mean(log_loss)
                return log_loss
            elif self.param.loss == 'mse':
                mse_loss = tf.reduce_mean((pred_u - y_true) ** 2)
                return mse_loss
            elif self.param.loss == 'mae':
                mae_loss = tf.reduce_mean(tf.abs(y_true - pred_u))
                return mae_loss
            else:
                sys.exit("'Loss type wrong! They can only be mae, mse or mve'")

        print(self.model.summary())
        self.model.compile(optimizer=self.optimizer, loss=loss_fn)

    def sample_batch(self, data_inputs, ground_truth, ruitu_inputs, batch_size, certain_id=None, certain_feature=None):
        max_i, _, max_j, _ = data_inputs.shape  # Example: (1148, 37, 10, 9)-(sample_ind, timestep, sta_id, features)

        id_ = np.random.randint(max_j, size=batch_size)
        i = np.random.randint(max_i, size=batch_size)
        batch_inputs = data_inputs[i, :, id_, :]
        batch_ouputs = ground_truth[i, :, id_, :]
        batch_ruitu = ruitu_inputs[i, :, id_, :]
        # id used for embedding
        if self.param.id_embd and (not self.param.time_embd):
            expd_id = np.expand_dims(id_, axis=1)
            batch_ids = np.tile(expd_id, (1, self.param.num_steps_to_predict))
            return batch_inputs, batch_ruitu, batch_ouputs, batch_ids
        elif (not self.param.id_embd) and (self.param.time_embd):
            time_range = np.array(range(self.param.num_steps_to_predict))
            batch_time = np.tile(time_range, (batch_size, 1))
            # batch_time = np.expand_dims(batch_time, axis=-1)

            return batch_inputs, batch_ruitu, batch_ouputs, batch_time
        elif (self.param.id_embd) and (self.param.time_embd):
            expd_id = np.expand_dims(id_, axis=1)
            batch_ids = np.tile(expd_id, (1, self.param.num_steps_to_predict))

            time_range = np.array(range(self.param.num_steps_to_predict))
            batch_time = np.tile(time_range, (batch_size, 1))
            # batch_time = np.expand_dims(batch_time, axis=-1)

            return batch_inputs, batch_ruitu, batch_ouputs, batch_ids, batch_time

        elif (not self.param.id_embd) and (not self.param.time_embd):
            return batch_inputs, batch_ruitu, batch_ouputs

    def fit(self, train_input_obs, train_input_ruitu, train_labels,
            val_input_obs, val_input_ruitu, val_labels, val_ids, val_times, batch_size,
            iterations=300, validation=True):

        print('Train batch size: {}'.format(batch_size))
        print('Validation on data size of {};'.format(val_input_obs.shape[0]))

        early_stop_count = 0

        model_json = self.model.to_json()
        with open(os.path.join(self.model_save_path, self.model_structure_name), "w") as json_file:
            json_file.write(model_json)
        print('Model structure has been saved at:', os.path.join(self.model_save_path, self.model_structure_name))

        for i in range(iterations):
            batch_inputs, batch_ruitu, batch_labels, batch_ids, batch_time = self.sample_batch(train_input_obs,
                                                                                               train_labels,
                                                                                               train_input_ruitu,
                                                                                               batch_size=batch_size)
            loss_ = self.model.train_on_batch(x=[batch_inputs, batch_ruitu, batch_ids, batch_time], y=[batch_labels])

            if (i + 1) % 50 == 0:
                print('Iteration:{}/{}. Training batch loss:{}'.
                      format(i + 1, iterations, loss_))

                if validation:
                    self.evaluate(val_input_obs, val_input_ruitu, val_labels, val_ids, val_times,
                                  each_station_display=False)
                    if len(self.val_loss_list) > 0:  # Early stopping
                        if (self.current_mean_val_loss) <= min(
                                self.val_loss_list):  # compare with the last early_stop_limit values except SELF
                            early_stop_count = 0
                            self.model.save_weights(os.path.join(self.model_save_path, self.model_weights_name))
                            print('The newest optimal model weights are updated at:',
                                  os.path.join(self.model_save_path, self.model_weights_name))
                        else:
                            early_stop_count += 1
                            print('Early-stop counter:', early_stop_count)
                    if early_stop_count == self.param.early_stop_limit:
                        self.EARLY_STOP = True
                        break

        print('###' * 10)
        if self.EARLY_STOP:
            print('Loading the best model before early-stop ...')
            self.model.load_weights(os.path.join(self.model_save_path, self.model_weights_name))

        print('Training finished! Detailed val loss with the best model weights:')
        self.evaluate(val_input_obs, val_input_ruitu, val_labels, val_ids, val_times, each_station_display=True)

    def evaluate(self, data_input_obs, data_input_ruitu, data_labels, data_ids, data_time, each_station_display=False):
        all_loss = []
        for i in range(self.param.num_stations):  # iterate for each station. (sample_ind, timestep, staionID, features)
            # batch_placeholders = np.zeros_like(data_labels[:,:,i,:])
            val_loss = self.model.evaluate(
                x=[data_input_obs[:, :, i, :], data_input_ruitu[:, :, i, :], data_ids[:, :, i], data_time],
                y=[data_labels[:, :, i, :]], verbose=False)

            all_loss.append(val_loss)

            if each_station_display:
                print('\tFor station 9000{}, val loss: {}'.format(i + 1, val_loss))

        self.current_mean_val_loss = np.mean(all_loss)
        print('Mean val loss:', self.current_mean_val_loss)

        self.val_loss_list.append(self.current_mean_val_loss)