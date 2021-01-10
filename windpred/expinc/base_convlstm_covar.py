import os

from windpred.utils.base import tag_path
from windpred.utils.model_base import DefaultConfig

from windpred.utils import exp_dir
from windpred.mhstn.base import get_covariates_history
from windpred.baseline import convlstm

from windpred.expinc.base import eval_mode

if __name__ == '__main__':
    tag = tag_path(os.path.abspath(__file__), 2)

    target = 'V'
    model_name = 'convlstm'

    if target == 'DIR':
        tag_file_list = [model_name]
        exp_dir.main_old('run', eval_mode, tag, tag_file_list)
        exp_dir.main_old('reduce', eval_mode, tag, tag_file_list)
    else:
        features_history = get_covariates_history(target)
        features_future = ['NEXT_NWP_{}'.format(target)]
        for mode in ['run', 'reduce']:
            convlstm.main(target, mode, eval_mode, DefaultConfig, tag, model_name, features_history, features_future)
