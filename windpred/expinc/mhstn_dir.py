from windpred.mhstn.mhstn import get_tags
from windpred.utils.exp_dir import main
from windpred.expinc.base import eval_mode


if __name__ == '__main__':
    file_exp_in = 'expinc_mhstn'  # !!! according to the eval_mode
    tag_file_list = get_tags()

    main('run', eval_mode, file_exp_in, tag_file_list)
    main('reduce', eval_mode, file_exp_in, tag_file_list)









