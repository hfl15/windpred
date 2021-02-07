import logging
import numpy as np

from .data_normlizer import DataNormalizer
from .data_loader import DataLoader
from .base import get_wind_threshold, split_index


def generate_data(x, y, period, window, target_size, start_idx, end_idx,
                  step=1, single_step=False, nwp_data=None, delay=True):
    """
    :return: 
        data = [primitive data, nwp predictions]
    """
    data = []
    targets = []

    if not delay:
        end_idx = end_idx if step >= period else end_idx-period
        for i in range(start_idx, end_idx, step):
            indices = range(i, i + period)
            data.append(x[indices])
            targets.append(y[indices])
    else:
        # limitation: window should be times of period to insure the of predictions range from 0 to period-1.
        start_idx = start_idx + window

        end_idx = end_idx - period

        for i in range(start_idx, end_idx, step):
            indices = range(i - window, i)
            d = x[indices]
            nwp = np.array([nwp_data[i + target_size - 1]] * window)
            if not single_step:
                nwp = np.array([nwp_data[i:i + target_size]] * (window // target_size))
            d = np.concatenate((d, nwp.reshape((x[indices].shape[0], nwp_data.shape[1]))), axis=1)
            data.append(d)

            if single_step:
                targets.append([y[i + target_size - 1]])
            else:
                targets.append(y[i:i + target_size])

    data, targets = np.array(data), np.array(targets)
    print("Finish to generate dataset: {0}\t{1}.".format(data.shape, targets.shape))

    return data, targets


class DataGenerator(object):
    def __init__(self, period, window, norm=None, path=None, x_divide_std=False):
        self.path = path
        self.period = period
        self.window = window
        self.norm = norm
        self.x_divide_std = x_divide_std

        self.loader = DataLoader(self.path)
        self.station_name = self.loader.get_station_name()
        self.df_all = self.loader.get_data()
        self.df_all['Month'] = self.df_all['DateTime'].apply(lambda x: x.year*100+x.month)
        self.month_start = self.df_all['Month'][0]
        self.month_end = self.df_all['Month'][self.df_all.shape[0]-1]
        self.month_list = np.unique(self.df_all['Month'])

        self.set_data()

    def set_data(self, month_list=None):
        if month_list is None:
            month_list = self.month_list
        self.df = self.df_all.copy()
        data_filter = [False] * self.df.shape[0]
        for m in month_list:
            data_filter = data_filter | (self.df['Month'] == m)
        self.df = self.df[data_filter]
        self.df_origin = self.df.copy()

        (self.train_start_idx, self.train_end_idx), (self.val_start_idx, self.val_end_idx),\
        (self.test_start_idx, self.test_end_idx) = split_index(self.df.shape[0], self.period)

        self.normalizer = None
        if self.norm is not None:
            self.normalizer = DataNormalizer(self.df, self.norm)
            self.df = self.normalizer.fit_transform(self.train_start_idx, self.train_end_idx)

        self.x_columns, self.y_columns = list(self.df.columns), list(self.df.columns)

    def prepare_data(self, target_size, train_step=1, test_step=1, single_step=False,
                        contains_pred=True):
        # add the predictions of nwp into feature set
        # these data will be filled in the following procedure, generate_data()
        nwp_columns = [col for col in self.df.columns if col.startswith('NWP')]
        nwp_data = self.df[nwp_columns].values
        if contains_pred:
            self.x_columns.extend(['NEXT_{}'.format(i) for i in nwp_columns])

        # prepare training data
        self.x_train, self.y_train = generate_data(
            self.df.values, self.df.values, self.period, self.window, target_size,
            self.train_start_idx, self.train_end_idx, step=train_step, single_step=single_step,
            nwp_data=nwp_data, delay=contains_pred)
        self.x_val, self.y_val = generate_data(
            self.df.values, self.df.values, self.period, self.window, target_size,
            self.val_start_idx, self.val_end_idx, step=train_step, single_step=single_step,
            nwp_data=nwp_data, delay=contains_pred)
        self.x_test, self.y_test = generate_data(
            self.df.values, self.df.values, self.period, self.window, target_size,
            self.test_start_idx, self.test_end_idx, step=test_step, single_step=single_step,
            nwp_data=nwp_data, delay=contains_pred)

        # prepare evaluation data
        self.x_eval = self.x_test
        _, self.y_eval = generate_data(
            self.df_origin.values, self.df_origin.values, self.period, self.window, target_size,
            self.test_start_idx, self.test_end_idx, step=test_step, single_step=single_step,
            nwp_data=nwp_data, delay=contains_pred)

    def extract_training_data(self, x_attributes, y_attributes):

        x_indices = []
        y_indices = []

        for attr in x_attributes:
            if attr in self.x_columns:
                x_indices.append(self.x_columns.index(attr))
            else:
                logging.error('Cannot find feature {0}.'.format(attr))

        for attr in y_attributes:
            if attr in self.y_columns:
                y_indices.append(self.y_columns.index(attr))
            else:
                logging.error('Cannot find feature {0}.'.format(attr))

        x_train, x_val, x_test = self.x_train[:, :, x_indices].astype(np.float32), \
                                 self.x_val[:, :, x_indices].astype(np.float32), \
                                 self.x_test[:, :, x_indices].astype(np.float32)

        y_train, y_val, y_test = self.y_train[:, :, y_indices].astype(np.float32), \
                                 self.y_val[:, :, y_indices].astype(np.float32), \
                                 self.y_test[:, :, y_indices]

        if self.x_divide_std is True and x_train.shape[-1] > 1:  # if the number of features is more than 1.
            std = np.mean(np.std(x_train.reshape((-1, x_train.shape[-1])), axis=1))
            x_train = x_train / std
            x_val = x_val / std
            x_test = x_test / std

        logging.debug('X_features selected: {0}, y_features selected: {1}'.format(x_attributes, y_attributes))

        logging.info('X_train shape:\t{0},\ty_train shape:\t{1}'.format(x_train.shape, y_train.shape))
        logging.info('X_val shape:\t{0},\ty_val shape:\t{1}'.format(x_val.shape, y_val.shape))
        logging.info('X_test shape:\t{0},\ty_test shape:\t{1}'.format(x_test.shape, y_test.shape))

        return (x_train, y_train.squeeze()), (x_val, y_val.squeeze()), (x_test, y_test.squeeze())

    def extract_evaluation_data(self, target):
        y_attributes = ['V', 'NWP_{}'.format(target), target]
        indices = [self.y_columns.index(attr) for attr in y_attributes]
        y_eval = self.y_eval[:, :, indices]

        speed = y_eval[:, :, 0].ravel()
        nwp = y_eval[:, :, 1].ravel()
        obs = y_eval[:, :, 2].ravel()
        filter_big_wind = (speed >= get_wind_threshold())

        return speed, nwp, obs, filter_big_wind


class DataGeneratorSpatial(DataGenerator):
    def __init__(self, period, window, norm=None, x_divide_std=False):
        super(DataGeneratorSpatial, self).__init__(
            period, window, norm=norm, path=None, x_divide_std=x_divide_std)

    def extract_evaluation_data(self, target):
        target_name, target_station = target.split('_')
        y_attributes = ['V_{}'.format(target_station), 'NWP_{}'.format(target_name), target]
        indices = [self.y_columns.index(attr) for attr in y_attributes]
        y_eval = self.y_eval[:, :, indices]

        speed = y_eval[:, :, 0].ravel()
        nwp = y_eval[:, :, 1].ravel()
        obs = y_eval[:, :, 2].ravel()
        filter_big_wind = (speed >= get_wind_threshold())

        return speed, nwp, obs, filter_big_wind



