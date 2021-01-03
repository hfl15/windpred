import os
import numpy as np
from multiprocessing import get_context

from windpred.utils.base import tag_path
from windpred.utils.model_base import DefaultConfig
from windpred.utils.exp import main_spatial_duq

from windpred.baseline.duq.main import run

from windpred.expinc.base import eval_mode


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

    ids_loss, ids_layers = np.meshgrid(range(len(loss_opts)), range(len(layers_opts)))
    ids_loss = ids_loss.ravel()
    ids_layers = ids_layers.ravel()
    params = [(loss_opts[iloss], layers_opts[ilay]) for iloss, ilay in zip(ids_loss, ids_layers)]

    n_processes = min(5, max(os.cpu_count() // 2, 1), len(params))
    with get_context("spawn").Pool(n_processes) as p:
        p.starmap(main, params)

