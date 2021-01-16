import os

from windpred.utils.model_base import DefaultConfig
from windpred.utils import exp_dir
from windpred.utils.exp import get_covariates_history_all, get_covariates_future_all
from windpred.baseline import convention


def _run(target, tag, eval_mode, features=None, with_spatial=False):
    if target == 'DIR':
        for model_name in convention.MODELS.keys():
            file_in = os.path.join(tag, model_name)
            tag_file_list = [None]
            for mode in ['run', 'reduce']:
                exp_dir.main(mode, DefaultConfig(), eval_mode, file_in, tag_file_list)
    else:
        for model_name in convention.MODELS.keys():
            for mode in ['run', 'reduce']:
                convention.main(tag, DefaultConfig(), target, mode, eval_mode, model_name,
                                features=features, with_spatial=with_spatial)


def run(target, tag, eval_mode):
    _run(target, tag, eval_mode)


def run_spatial(target, tag, eval_mode):
    _run(target, tag, eval_mode, with_spatial=True)


def run_spatial_covar_all(target, tag, eval_mode):
    features_history = get_covariates_history_all()
    features_future = get_covariates_future_all()
    features = features_history + features_future
    _run(target, tag, eval_mode, features, with_spatial=True)

