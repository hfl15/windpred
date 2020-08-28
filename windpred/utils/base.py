import os
import math
import numpy as np
from scipy import stats

import matplotlib.pyplot as plt


DIR_LOG = '../cache/'
DATA_DIR = '../../data/20200107update_csv'
NWP_DIR = '../../data/nwp'


"""
    For tf
"""


def get_tf_keras():
    """
    reference: Feb 7 2020, https://medium.com/@starriet87/tensorflow-2-0-wanna-limit-gpu-memory-10ad474e2528
    :return:
    """
    import tensorflow as tf
    if tf.__version__.startswith('1.'):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        K = tf.keras.backend
        K.set_session(sess)
    else:  # tf2.x
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                print(e)
        K = tf.keras.backend
    return tf, K


"""
    For path and dir related
"""


def path_split(path):
    folders = []
    while 1:
        path, folder = os.path.split(path)

        if folder != "":
            folders.append(folder)
        else:
            if path != "":
                folders.append(path)
            break
    folders.reverse()
    return folders


def tag_path(path, nback=1):
    """
    example:
        tag_path(os.path.abspath(__file__), 1) # return file name
    :param path: 
    :param nback: 
    :return: 
    """
    folders = path_split(path)
    nf = len(folders)

    assert nback >= 1, "nback={} should be larger than 0.".format(nback)
    assert nback <= nf, "nback={} should be less than the number of folder {}!".format(nback, nf)

    tag = folders[-1].split('.')[0]
    if nback > 0:
        for i in range(2, nback + 1):
            tag = folders[-i] + '_' + tag
    return tag


def make_dir(path):
    if os.path.exists(path) is False:
        os.makedirs(path)


"""
    For visualization
"""


def plot_and_save_comparison(y, y_pred, dir_log, filename='compare.png', title=None):
    plt.plot(y, '.-', label='truth')
    plt.plot(y_pred, '.-', label='pred')
    plt.legend(loc='best')
    if title is not None:
        plt.title(title)
    plt.savefig(os.path.join(dir_log, filename))
    plt.close()


def plot_train_valid_loss(loss, val_loss, dir_log, title='Training_Validation_Loss'):
    epochs = range(len(loss))
    plt.figure()
    plt.plot(epochs, loss, 'b', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title(title)
    plt.legend(loc='best')
    plt.savefig(os.path.join(dir_log, '{}.png'.format(title)))
    plt.close()


"""
    For params and settings
"""


def get_wind_threshold():
    return 3


def get_missing_tag():
    return -9999


def angle_to_radian(angle):
    return angle / (2 * math.pi)


def uv_to_degree(u, v):
    """
        u = -spd*sin(degree/360*2*pi)
        v = -spd*cos(degree/360*2*pi)
    :param u:
    :param v:
    :return:
    """
    degree = np.arctan(u / v) / np.pi * 180
    if u < 0 and v < 0:
        degree = degree
    elif u >= 0 and v < 0:
        degree = degree + 360
    else:  # (u<0 and v>=0) or (u>=0 and v>=0)
        degree = degree + 180

    return degree


def uv_to_degree_vec(u, v):
    assert len(u) == len(v), \
        "len(u)={} should be equal to len(v)={}".format(len(u), len(v))
    return np.array([uv_to_degree(u[i], v[i]) for i in range(len(u))])


def uv_to_speed(u, v):
    return np.sqrt(u*2, v*2)


"""
    For dataset parser
"""


def get_files_path():
    data_path_list = [os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR) if os.path.isfile(os.path.join(DATA_DIR, f))]
    data_path_list = sorted(data_path_list)
    nwp_path = os.path.join(NWP_DIR, 'ec.csv')

    return data_path_list, nwp_path


def get_station_name(path):
    return os.path.basename(path).split('.')[0]


"""
   For data pre-processing
"""


def fill_missing(df):
    new_df = df
    missing_detail = {}
    missing_tag = get_missing_tag()
    for col in df:
        dfc = df[col]
        if str(dfc.dtypes) == 'object':
            continue

        missing_indices = np.where(dfc.values == missing_tag)[0]
        missing_detail[col] = len(missing_indices)

        for i in missing_indices:
            l = i - 1
            while l in missing_indices and l > 0:
                l = l - 1
            r = i + 1
            while r in missing_indices and r < len(dfc.values) - 1:
                r = r + 1
            dfc[i] = (dfc[l] + dfc[r]) / 2
        new_df[col] = dfc

    print(missing_detail)

    return new_df


def split_index(n_samples, period):
    n_test = 30 * period  # according to the number of days in a month

    # [start_index, end_index)
    train_start_idx, train_end_idx = 0, n_samples - 2 * n_test
    val_start_idx, val_end_idx = n_samples - 2 * n_test, n_samples - n_test
    test_start_idx, test_end_idx = n_samples - n_test, n_samples

    print("Finish to split index. train: [{0}, {1}), val: [{2}, {3}), test: [{4}, {5})".format(
        train_start_idx, train_end_idx, val_start_idx, val_end_idx, test_start_idx, test_end_idx
    ))

    return (train_start_idx, train_end_idx), (val_start_idx, val_end_idx), (test_start_idx, test_end_idx)


def get_outliers(samples, threshold=0.05):
    """ useless
    :param threshold: 0.05
    :param samples: np.array
    :return: outliers: np.array
    """
    mu = np.mean(samples)
    sigma = np.std(samples)
    prob = stats.norm.pdf(samples, mu, sigma)
    prob_mu = np.mean(prob)
    prob_sigma = np.std(prob)
    prob_prob = stats.norm.pdf(prob, prob_mu, prob_sigma)

    return np.where(prob_prob < threshold)[0]

