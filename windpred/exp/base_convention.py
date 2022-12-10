import os

from windpred.utils.model_base import DefaultConfig
from windpred.utils import exp_dir
from windpred.mhstn.base import get_covariates_history
from windpred.utils.exp import get_covariates_history_all, get_covariates_future_all
from windpred.baseline import convention


def _run(target, tag, model_name, eval_mode, features=None, with_spatial=False):
    if target == 'DIR':
        file_in = os.path.join(tag, model_name)
        tag_file_list = [None]
        for mode in ['run', 'reduce']:
            exp_dir.main(mode, DefaultConfig(), eval_mode, file_in, tag_file_list)
    else:
        for mode in ['run', 'reduce']:
            convention.main(tag, DefaultConfig(), target, mode, eval_mode, model_name,
                            features=features, with_spatial=with_spatial)


def run(target, tag, model_name, eval_mode):
    _run(target, tag, model_name, eval_mode)


def run_h(target, tag, model_name, eval_mode):
    features = [target]
    _run(target, tag, model_name, eval_mode, features)


def run_f(target, tag, model_name, eval_mode):
    features = ['NEXT_NWP_{}'.format(target)]
    _run(target, tag, model_name, eval_mode, features)


def run_spatial(target, tag, model_name, eval_mode):
    """It is time consuming."""
    _run(target, tag, model_name, eval_mode, with_spatial=True)


def run_spatial_covar(target, tag, model_name, eval_mode):
    """It is time consuming."""
    if target == 'DIR':
        features_history = features_future = [None]
    else:
        features_history = get_covariates_history(target)
        features_future = ['NEXT_NWP_{}'.format(target)]
    features = features_history + features_future
    _run(target, tag, model_name, eval_mode, features, with_spatial=True)


def run_spatial_covar_all(target, tag, model_name, eval_mode):
    """It is time consuming."""
    features_history = get_covariates_history_all()
    features_future = get_covariates_future_all()
    features = features_history + features_future
    _run(target, tag, model_name, eval_mode, features, with_spatial=True)


run_temporal_funcs = {'history_future': run,
                      'history': run_h,
                      'future': run_f}

