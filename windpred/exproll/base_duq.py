import os

from windpred.utils.base import tag_path
from windpred.utils.model_base import DefaultConfig
from windpred.utils.exp import main_spatial_duq

from windpred.baseline.duq.main import run


from windpred.exproll.base import eval_mode


if __name__ == '__main__':
    tag = tag_path(os.path.abspath(__file__), 2)
    config = DefaultConfig()

    mode = 'run'
    target = 'V'
    features_history, features_future = [target], ['NEXT_NWP_{}'.format(target)]
    csv_result_list = ['metrics_model.csv', 'metrics_nwp.csv']

    func = run(features_history, features_future)
    main_spatial_duq(target, mode, eval_mode, DefaultConfig(), tag, func, csv_result_list)



