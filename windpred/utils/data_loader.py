import logging

import math
import pandas as pd
import numpy as np

from .base import get_station_name, fill_missing, get_files_path


def load_nwp_data():
    _, nwp_path = get_files_path()
    print('Loading nwp data from: {0}'.format(nwp_path))
    nwp_data = pd.read_csv(nwp_path)
    columns = list(nwp_data.columns)
    columns.remove('DateTime')
    nwp_data[columns] = nwp_data[columns].astype(np.float)
    nwp_data['DateTime'] = pd.to_datetime(nwp_data['DateTime'])

    nwp_data['DIRRadian'] = nwp_data['DIR'] / 360 * (2 * math.pi)

    nwp_data.columns = ['NWP_{}'.format(i) for i in nwp_data.columns]

    print("Finish to load nwp data: ", nwp_data.shape)
    logging.info('NWP_DIR describe:\n', nwp_data['NWP_DIR'].describe())

    return nwp_data


def load_wind_data(path):
    print('Loading data from: {0}'.format(path))
    wind_data = pd.read_csv(path)
    wind_data['DateTime'] = pd.to_datetime(wind_data['DateTime'])
    columns = list(wind_data.columns)
    columns.remove('DateTime')
    wind_data[columns] = wind_data[columns].astype(np.float)

    wind_data = fill_missing(wind_data)

    # add some columns
    wind_data['DIRRadian'] = wind_data['DIR'] / 360 * (2 * math.pi)
    wind_data['VX'] = -np.sin(wind_data['DIRRadian']) * wind_data['V']
    wind_data['VY'] = -np.cos(wind_data['DIRRadian']) * wind_data['V']
    # wind_data = wind_data.drop(['DIRRadian'], axis=1)

    print("Finish to load wind data: ", wind_data.shape, wind_data['DIR'].shape)
    logging.info('DIR describe:\n', wind_data['DIR'].describe())

    return wind_data


def load_wind_all_stations():
    data_paths, _ = get_files_path()

    # init 0
    wind_data_all = load_wind_data(data_paths[0])

    for i in range(len(data_paths)):
        wind_data = load_wind_data(data_paths[i])
        renamed_columns = ['{}_S{}'.format(x, i) for x in wind_data.columns]
        wind_data.columns = renamed_columns

        wind_data_all = pd.merge(left=wind_data_all, right=wind_data,
                                 left_on='DateTime', right_on='DateTime_S{}'.format(i))

    return wind_data_all


def load_data(path):
    wind_data = load_wind_data(path)
    nwp_data = load_nwp_data()
    data = pd.merge(left=wind_data, right=nwp_data, left_on='DateTime', right_on='NWP_DateTime')
    return data


def load_data_all():
    wind_data = load_wind_all_stations()
    nwp_data = load_nwp_data()
    data = pd.merge(left=wind_data, right=nwp_data, left_on='DateTime', right_on='NWP_DateTime')
    return data


class DataLoader(object):
    def __init__(self, path=None):
        self.path = path
        if self.path is not None:
            self._data = load_data(self.path)
            self._station_name = get_station_name(self.path)
        else:  # load all data
            self._data = load_data_all()
            self._station_name = 'multiple'

    def get_station_name(self):
        return self._station_name

    def get_data(self):
        return self._data
