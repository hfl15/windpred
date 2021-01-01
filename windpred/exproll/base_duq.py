import os
import numpy as np
from multiprocessing import get_context

from windpred.utils.base import tag_path
from windpred.utils.model_base import DefaultConfig
from windpred.utils.exp import main_spatial_duq

from windpred.baseline.duq.main import run

from windpred.exproll.base import eval_mode


def main(loss, layers):
    tag_g = tag_path(os.path.abspath(__file__), 2)
    target = 'V'
    mode_opts = ['run', 'reduce', 'clear']
    features_history, features_future = [target], ['NEXT_NWP_{}'.format(target)]
    csv_result_list = ['metrics_model.csv', 'metrics_nwp.csv']
    for mode in mode_opts:
        func = run(features_history, features_future, loss=loss, layers=layers)
        layers_str = [str(l) for l in layers]
        tag = tag_g + '_' + '{}-{}'.format(loss, "-".join(layers_str))
        main_spatial_duq(target, mode, eval_mode, DefaultConfig(), tag, func, csv_result_list)


if __name__ == '__main__':
    loss_opts = ['mae', 'mse', 'mve']
    layers_opts = [[32], [32, 32], [50], [50, 50], [200], [200, 200], [300], [300, 300]]

    params_finished = [
        ('mae', [32]),
        ('mae', [32, 32]),
        ('mae', [50]),
        ('mae', [50, 50]),
        ('mae', [200]),
        ('mse', [32]),
        ('mse', [32, 32]),
        ('mse', [50]),
        ('mse', [50, 50]),
        ('mse', [200]),
        ('mve', [32]),
        ('mve', [32, 32]),
        ('mve', [50]),
        ('mve', [50, 50])
    ]

    ids_loss, ids_layers = np.meshgrid(range(len(loss_opts)), range(len(layers_opts)))
    ids_loss = ids_loss.ravel()
    ids_layers = ids_layers.ravel()

    n_processes = min(max(os.cpu_count()//2, 1), len(ids_loss))
    with get_context("spawn").Pool(n_processes) as p:
        params = [(loss_opts[iloss], layers_opts[ilay]) for iloss, ilay in zip(ids_loss, ids_layers)]
        for par in params_finished:
            params.remove(par)
        p.starmap(main, params)



