import os

from windpred.utils.base import tag_path
from windpred.utils.model_base import DefaultConfig
from windpred.utils.exp import get_covariates_history
from windpred.baseline.convlstm import main

from windpred.expinc.base import eval_mode

if __name__ == '__main__':
    tag = tag_path(os.path.abspath(__file__), 2)

    target = 'SPD10'
    features_history = get_covariates_history(target)
    features_future = ['NEXT_NWP_{}'.format(target)]
    mode = 'run'
    model_name = 'convlstm'
    main(target, mode, eval_mode, DefaultConfig, tag, model_name, features_history, features_future)
