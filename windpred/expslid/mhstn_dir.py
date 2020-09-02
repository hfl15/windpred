from windpred.mhstn.mhstn import get_tags
from windpred.expslid.dir import main
from windpred.expslid.base import eval_mode


if __name__ == '__main__':
    file_exp_in = 'expslid_mhstn'
    tag_file_list = get_tags()

    main('run', eval_mode, file_exp_in, tag_file_list)
    main('reduce', eval_mode, file_exp_in, tag_file_list)









