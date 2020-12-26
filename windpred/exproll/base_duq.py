import os
import numpy as np
from multiprocessing import Pool

from windpred.utils.base import tag_path
from windpred.utils.model_base import DefaultConfig
from windpred.utils.exp import main_spatial_duq

from windpred.baseline.duq.main import run


from windpred.exproll.base import eval_mode


def main(loss, layers):
    tag = tag_path(os.path.abspath(__file__), 2)
    target = 'V'
    mode_opts = ['run', 'reduce', 'clear']
    features_history, features_future = [target], ['NEXT_NWP_{}'.format(target)]
    csv_result_list = ['metrics_model.csv', 'metrics_nwp.csv']
    for mode in mode_opts:
        func = run(features_history, features_future, loss=loss, layers=layers)
        tag = tag + '_' + mode.split('_')[-1]
        main_spatial_duq(target, mode, eval_mode, DefaultConfig(), tag, func, csv_result_list)


if __name__ == '__main__':
    # loss_opts = ['mae', 'mse', 'mve']
    # layers_opts = [[32], [32, 32], [50], [50, 50], [200], [200, 200], [300], [300, 300]]
    loss_opts = ['mae']
    layers_opts = [[32], [32, 32]]
    
    ids_loss, ids_layers = np.meshgrid(range(len(loss_opts)), range(len(layers_opts)))
    ids_loss = ids_loss.ravel()
    ids_layers = ids_layers.ravel()

    n_processes = min(max(os.cpu_count()-2, 1), len(ids_loss))
    with Pool(n_processes) as p:
        p.starmap(main, [(loss_opts[iloss], layers_opts[ilay]) for iloss, ilay in zip(ids_loss, ids_layers)])


    # from IPython import embed; embed()
    #
    # mode = 'clear_mae-50-50'

    # if mode.endswith('mae-50-50'):
    #     func = run(features_history, features_future, loss='mae', layers=[50, 50])
    # elif mode.endswith('mse-50-50'):
    #     func = run(features_history, features_future, loss='mse', layers=[50, 50])
    # elif mode.endswith('mve-50-50'):
    #     func = run(features_history, features_future, loss='mve', layers=[50, 50])
    # else:
    #     raise ValueError('mode={} can not be found!'.format(mode))

    # func = run(features_history, features_future, loss='mve', layers=[32])
    #
    # # tag = tag + '_' + mode.split('_')[-1]
    # tag = tag + mode.split('_')[-1]
    # main_spatial_duq(target, mode, eval_mode, DefaultConfig(), tag, func, csv_result_list)



