import numpy as np

from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, mean_absolute_error

EPSILON = 1e-07


def get_normalized_factor(y_true, mode):
    if mode == "mean":
        normalized_factor = np.mean(y_true)
    elif mode == "max":
        normalized_factor = np.max(y_true)
    elif mode == "std":
        normalized_factor = np.std(y_true)
    elif mode == "diff-max-min":
        normalized_factor = np.max(y_true) - np.min(y_true)
    elif mode.startswith('quantile'):
        # inter quartile range
        # e.g., 'quantile-0.75-0.25' means Q3-Q1. Q1=0.25, Q2=0.5, Q3=0.75.
        tag, q_high, q_low = mode.split('-')
        q_high, q_low = np.float(q_high), np.float(q_low)
        q_values = np.quantile(y_true, [q_high, q_low])
        normalized_factor = q_values[0] - q_values[1]
    else:
        normalized_factor = None
        ValueError("mode={} can not be found!".format(mode))

    return normalized_factor


def mse(y_true, y_pred):
    """Mean Square Error"""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    ret = np.mean((y_true - y_pred)**2)
    return ret


def rmse(y_true, y_pred):
    """Root Mean Squared Error"""
    ret = mse(y_true, y_pred)
    return np.sqrt(ret)


def mae(y_true, y_pred):
    """Mean Absolute Error"""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.mean(np.abs(y_true - y_pred))


def nmae(y_true, y_pred, mode='mean'):
    """Normalized Mean Absolute Error"""
    normalized_factor = get_normalized_factor(y_true, mode)
    ret = mae(y_true, y_pred) / normalized_factor
    # if mode = 'mean', nmae = np.sum(np.abs(y_true - y_pred)) / np.sum(y_true)
    return ret


def mape(y_true, y_pred):
    """Mean Absolute Percentage Error"""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    ret = np.mean(np.abs(y_true-y_pred) / np.maximum(EPSILON, np.abs(y_true)))
    return ret


def mpe(y_true, y_pred):
    """Mean Percentage Error"""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    ret = np.mean((y_true-y_pred) / np.maximum(EPSILON, y_true))
    return ret


def rae(y_true, y_pred):
    """Relative Absolute Error"""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    y_mean = np.mean(y_true)
    ret = np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true - y_mean))
    return ret


def rse(y_true, y_pred):
    """Relative Squared Error"""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    y_mean = np.mean(y_true)
    ret = np.sum((y_true - y_pred)**2) / np.sum((y_true - y_mean)**2)
    # that is equal to: mse(y_true, y_pred) / np.mean((y_true - y_mean)**2)
    return ret


def rrse(y_true, y_pred):
    """Root Relative Squared Error"""
    return np.sqrt(rse(y_true, y_pred))


def rrmse(y_true, y_pred):
    """Relative Root Mean Squared Error"""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    ret = rmse(y_true, y_pred) / np.sqrt(np.sum(y_true**2))
    return ret


def nrmse(y_true, y_pred, mode='mean'):
    """Normalized Root Mean Squared Error"""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    normalized_factor = get_normalized_factor(y_true, mode)
    ret = rmse(y_true, y_pred) / normalized_factor

    return ret


if __name__ == '__main__':
    print("Some test cases: ")
    y_true = [1, 10, 1e6]
    y_pred = [0.9, 15, 1.2e6]

    print("mae", mae(y_true, y_pred), mean_absolute_error(y_true, y_pred))
    print("nmae", nmae(y_true, y_pred))
    print("mape", mape(y_true, y_pred), mean_absolute_percentage_error(y_true, y_pred))
    print("mpe", mpe(y_true, y_pred))
    print("rae", rae(y_true, y_pred))
    print("rse", rse(y_true, y_pred))
    print("rrse", rrse(y_true, y_pred))
    print("rrmse", rrmse(y_true, y_pred))
    print("nrmse", nrmse(y_true, y_pred))
    print("nrmse, quantile: ", nrmse(y_true, y_pred, mode="quantile-0.75-0.25"))

    y_true = [3, -0.5, 2, 7]
    y_pred = [2.5, 0.0, 2, 8]
    print("mse", mse(y_true, y_pred), mean_squared_error(y_true, y_pred))
    print("rmse", rmse(y_true, y_pred), mean_squared_error(y_true, y_pred, squared=False))

