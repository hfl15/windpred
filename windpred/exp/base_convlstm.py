from windpred.utils.model_base import DefaultConfig
from windpred.mhstn.base import get_covariates_history
from windpred.baseline import convlstm
from windpred.utils import exp_dir


def _run(target, tag, eval_mode, features_history, features_future):
    model_name = 'convlstm'

    if target == 'DIR':
        tag_file_list = [model_name]
        for mode in ['run', 'reduce']:
            exp_dir.main(mode, DefaultConfig(), eval_mode, tag, tag_file_list)
    else:
        for mode in ['run', 'reduce']:
            convlstm.main(target, mode, eval_mode, DefaultConfig(), tag, model_name, features_history, features_future)


def run(target, tag, eval_mode):
    features_history, features_future = [target], ['NEXT_NWP_{}'.format(target)]
    _run(target, tag, eval_mode, features_history, features_future)


def run_covar(target, tag, eval_mode):
    if target == 'DIR':
        features_history = features_future = [None]
    else:
        features_history = get_covariates_history(target)
        features_future = ['NEXT_NWP_{}'.format(target)]
    _run(target, tag, eval_mode, features_history, features_future)
