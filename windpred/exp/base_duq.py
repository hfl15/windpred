"""
    Run the model with the best settings.
"""
import os
import numpy as np
from multiprocessing import get_context

from windpred.utils.exp import get_covariates_history_all, get_covariates_future_all
from windpred.mhstn.base import get_covariates_history
from windpred.utils.model_base import DefaultConfig
from windpred.utils import exp_dir
from windpred.utils.exp import main_spatial_duq

from windpred.baseline.duq.main import run as duqrun

"""
    Run the model with the best settings.
"""


def _run_best(target, tag, eval_mode, features_history, features_future):
    if target == 'DIR':
        tag_file_list = [None]
        for mode in ['run', 'reduce']:
            exp_dir.main(mode, DefaultConfig(), eval_mode, tag, tag_file_list)
    else:
        loss = 'mve'
        layers = [32]
        csv_result_list = ['metrics_model.csv', 'metrics_nwp.csv']
        func = duqrun(features_history, features_future, loss=loss, layers=layers)
        for mode in ['run', 'reduce']:
            main_spatial_duq(target, mode, eval_mode, DefaultConfig(), tag, func, csv_result_list)


def run_best(target, tag, eval_mode):
    features_history, features_future = [target], ['NEXT_NWP_{}'.format(target)]
    _run_best(target, tag, eval_mode, features_history, features_future)


def run_best_covar(target, tag, eval_mode):
    if target == 'DIR':
        features_history = features_future = [None]
    else:
        features_history = get_covariates_history(target)
        features_future = ['NEXT_NWP_{}'.format(target)]
    _run_best(target, tag, eval_mode, features_history, features_future)


def run_best_covar_all(target, tag, eval_mode):
    features_history = get_covariates_history_all()
    features_future = get_covariates_future_all()
    _run_best(target, tag, eval_mode, features_history, features_future)


"""
    Try a batch of hyper-parameters that used in the primitive paper.
"""


def _run_grid_search(loss, layers, target, tag, eval_mode):
    mode_opts = ['run', 'reduce', 'clear']
    features_history, features_future = [target], ['NEXT_NWP_{}'.format(target)]
    csv_result_list = ['metrics_model.csv', 'metrics_nwp.csv']
    for mode in mode_opts:
        func = duqrun(features_history, features_future, loss=loss, layers=layers)
        layers_str = [str(l) for l in layers]
        tag_curr = tag + '_' + '{}-{}'.format(loss, "-".join(layers_str))
        main_spatial_duq(target, mode, eval_mode, DefaultConfig(), tag_curr, func, csv_result_list)


def run_grid_search(target, tag, eval_mode):
    loss_opts = ['mae', 'mse', 'mve']
    layers_opts = [[32], [32, 32], [50], [50, 50], [200], [200, 200], [300], [300, 300]]

    ids_loss, ids_layers = np.meshgrid(range(len(loss_opts)), range(len(layers_opts)))
    ids_loss = ids_loss.ravel()
    ids_layers = ids_layers.ravel()
    params = [(loss_opts[iloss], layers_opts[ilay], target, tag, eval_mode)
              for iloss, ilay in zip(ids_loss, ids_layers)]

    n_processes = min(5, max(os.cpu_count() // 2, 1), len(params))
    with get_context("spawn").Pool(n_processes) as p:
        p.starmap(_run_grid_search, params)


