import os

from windpred.utils.base import tag_path
from windpred.utils.model_base import DefaultConfig
from windpred.utils import exp_dir

from windpred.mhstn.base import CSV_RESULT_FILES
from windpred.mhstn import mhstn
from windpred.expslid.base import eval_mode


if __name__ == '__main__':
    tag = tag_path(os.path.abspath(__file__), 2)

    target = 'DIR10'

    if target == 'DIR10':
        tag_file_list = mhstn.get_tags()
        exp_dir.main('run', eval_mode, tag, tag_file_list)
        exp_dir.main('reduce', eval_mode, tag, tag_file_list)
    else:
        features_history = ['SPD10', 'U10', 'V10', 'DIRRadian', 'SLP', 'T2', 'RH2']
        features_future = ['NEXT_NWP_{}'.format(feat) for feat in
                           ['SPD10', 'U10', 'V10', 'DIRRadian', 'SLP', 'T2', 'RH2']]
        mode = 'temporal'
        csv_result_list = CSV_RESULT_FILES
        mhstn.main(target, mode, eval_mode, DefaultConfig, tag, features_history, features_future, csv_result_list)



