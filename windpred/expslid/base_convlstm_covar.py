import os

from windpred.utils.base import tag_path
from windpred.utils.model_base import DefaultConfig
from windpred.utils.exp import get_covariates_history
from windpred.utils import exp_dir

from windpred.baseline import convlstm

from windpred.expslid.base import eval_mode

if __name__ == '__main__':
    tag = tag_path(os.path.abspath(__file__), 2)

    target = 'DIR10'
    model_name = 'convlstm'

    if target == 'DIR10':
        tag_file_list = [model_name]
        exp_dir.main('run', eval_mode, tag, tag_file_list)
        exp_dir.main('reduce', eval_mode, tag, tag_file_list)
    else:
        features_history = get_covariates_history(target)
        features_future = ['NEXT_NWP_{}'.format(target)]
        mode = 'run'
        convlstm.main(target, mode, eval_mode, DefaultConfig, tag, model_name, features_history, features_future)

