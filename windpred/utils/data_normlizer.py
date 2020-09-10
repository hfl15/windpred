import logging
import numpy as np
from pandas import DataFrame
from sklearn.preprocessing import *


def get_norm_method(method):
    if method.lower() == 'standard':
        return StandardScaler()
    elif method.lower() == 'submean':
        return StandardScaler(with_std=False)
    elif method.lower() == 'minmax':
        return MinMaxScaler()
    elif method.lower() == 'maxabs':
        return MaxAbsScaler()
    else:
        logging.error("Unexpected normalization method: {0}.".format(method))


class DataNormalizer(object):
    def __init__(self, data, norm_method):
        """
        :param data: data-frame format
        :param norm_method:
        """
        self.data = data
        self.norm_method = norm_method
        self.scaler_dict = {}
        self._init_scaler_dict()

    def fit_transform(self, train_start_idx, train_end_idx):
        train_data = self.data[train_start_idx: train_end_idx]
        norm_data = DataFrame()

        for col in train_data.columns:
            if col in self.scaler_dict.keys():
                self.scaler_dict[col].fit(train_data[col].values.reshape(-1, 1))
                norm_data[col] = self.scaler_dict[col].transform(self.data[col].values.reshape(-1, 1)).ravel()
            else:
                logging.warning("Column {0} is ignored during normalization!".format(col))
                norm_data[col] = self.data[col]

        return norm_data

    def inverse_transform(self, feature, target_data):
        if feature in self.scaler_dict.keys():
            return self.scaler_dict[feature].inverse_transform(target_data.reshape(-1, 1)).ravel()
        else:
            logging.error("Feature {0} is not found!".format(target_data))

    def _init_scaler_dict(self):
        for col in self.data.columns:
            if self.data[col].dtype == np.float or  self.data[col].dtype == np.int:
                self.scaler_dict[col] = get_norm_method(self.norm_method)

