from windpred.exp.base_duq import run_best_covar_all
from windpred.expinc.base import *


if __name__ == '__main__':
    tag = get_tag(__file__)
    target = 'V'
    run_best_covar_all(target, tag, eval_mode)

