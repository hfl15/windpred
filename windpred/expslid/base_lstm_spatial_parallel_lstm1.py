import os

from windpred.utils.base import tag_path
from windpred.utils.model_base import DefaultConfig

from windpred.utils.exp import main_spatial
from windpred.baseline.lstm_spatial import run_lstm

from windpred.expslid.base import eval_mode


if __name__ == '__main__':
    tag = tag_path(os.path.abspath(__file__), 2)

    target = 'SPD10'
    features_history, features_future = [target], ['NEXT_NWP_{}'.format(target)]
    mode = 'run'
    concat_mode = 'parallel'
    model_mode = 'lstm1'
    csv_result_list = ['metrics_model.csv', 'metrics_nwp.csv']

    func = run_lstm(features_history, features_future, concat_mode, model_mode)
    main_spatial(target, mode, eval_mode, DefaultConfig, tag, func, csv_result_list)




