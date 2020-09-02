import os

from windpred.utils.base import path_split
from windpred.utils.exp_dir import main
from windpred.mhstn.mhstn import get_tags
from windpred.expslid.base import eval_mode


if __name__ == '__main__':
    file_exp_in = '{}_mhstn'.format(path_split(os.path.abspath(__file__))[-2])
    tag_file_list = get_tags()

    main('run', eval_mode, file_exp_in, tag_file_list)
    main('reduce', eval_mode, file_exp_in, tag_file_list)









