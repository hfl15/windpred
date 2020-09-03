import os

from windpred.utils.base import tag_path
from windpred.utils.model_base import DefaultConfig
from windpred.utils.exp import get_covariates_history

from windpred.mhstn.base import CSV_RESULT_FILES
from windpred.mhstn.mhstn import main

from windpred.expinc.base import eval_mode


if __name__ == '__main__':
    tag = tag_path(os.path.abspath(__file__), 2)

    target = 'SPD10'
    features_history = get_covariates_history(target)
    features_future = ['NEXT_NWP_{}'.format(target)]

    mode = 'temporal'

    csv_result_list = CSV_RESULT_FILES

    main(target, mode, eval_mode, DefaultConfig, tag, features_history, features_future, csv_result_list)









