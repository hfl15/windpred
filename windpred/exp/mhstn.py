from windpred.utils.model_base import DefaultConfig
from windpred.utils.exp import get_covariates_history_all, get_covariates_future_all
from windpred.utils import exp_dir

from windpred.mhstn.base import CSV_RESULT_FILES, get_covariates_history
from windpred.mhstn import mhstn


def _run(target, tag, eval_mode, features_history, features_future):
    if target == 'DIR':
        tag_file_list = mhstn.get_tags()
        for mode in ['run', 'reduce']:
            exp_dir.main(mode, DefaultConfig(), eval_mode, tag, tag_file_list)
    else:
        mode_list = ['temporal', 'spatial-conv', 'combine-conv', 'reduce']
        for mode in mode_list:
            csv_result_list = CSV_RESULT_FILES
            mhstn.main(target, mode, eval_mode, DefaultConfig, tag, features_history, features_future, csv_result_list)


def run(target, tag, eval_mode):
    features_history, features_future = [target], ['NEXT_NWP_{}'.format(target)]
    _run(target, tag, eval_mode, features_history, features_future)
    # if target == 'DIR':
    #     tag_file_list = mhstn.get_tags()
    #     for mode in ['run', 'reduce']:
    #         exp_dir.main(mode, DefaultConfig(), eval_mode, tag, tag_file_list)
    # else:
    #     features_history, features_future = [target], ['NEXT_NWP_{}'.format(target)]
    #     mode_list = ['temporal', 'spatial-conv', 'combine-conv', 'reduce']
    #     for mode in mode_list:
    #         csv_result_list = CSV_RESULT_FILES
    #         mhstn.main(target, mode, eval_mode, DefaultConfig, tag, features_history, features_future, csv_result_list)


def run_covar(target, tag, eval_mode):
    features_history = get_covariates_history(target)
    features_future = ['NEXT_NWP_{}'.format(target)]
    _run(target, tag, eval_mode, features_history, features_future)
    # if target == 'DIR':
    #     tag_file_list = mhstn.get_tags()
    #     exp_dir.main('run', eval_mode, tag, tag_file_list)
    #     exp_dir.main_old('reduce', eval_mode, tag, tag_file_list)
    # else:
    #     features_history = get_covariates_history(target)
    #     features_future = ['NEXT_NWP_{}'.format(target)]
    #     mode_list = ['temporal', 'spatial-conv', 'combine-conv', 'reduce']
    #     for mode in mode_list:
    #         csv_result_list = CSV_RESULT_FILES
    #         mhstn.main(target, mode, eval_mode, DefaultConfig, tag, features_history, features_future,
    #                    csv_result_list)


def run_covar_all(target, tag, eval_mode):
    features_history = get_covariates_history_all()
    features_future = get_covariates_future_all()
    _run(target, tag, eval_mode, features_history, features_future)
    # if target == 'DIR':
    #     tag_file_list = mhstn.get_tags()
    #     exp_dir.main_old('run', eval_mode, tag, tag_file_list)
    #     exp_dir.main_old('reduce', eval_mode, tag, tag_file_list)
    # else:
    #     features_history = get_covariates_history_all()
    #     features_future = get_covariates_future_all()
    #     mode_list = ['temporal', 'spatial-conv', 'combine-conv', 'reduce']
    #     for mode in mode_list:
    #         csv_result_list = CSV_RESULT_FILES
    #         mhstn.main(target, mode, eval_mode, DefaultConfig, tag, features_history, features_future, csv_result_list)






