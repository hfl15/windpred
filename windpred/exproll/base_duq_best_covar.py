from windpred.exp.base_duq import run_best_covar
from windpred.exproll.base import *


if __name__ == '__main__':
    tag = get_tag(__file__)
    target = 'V'
    run_best_covar(target, tag, eval_mode)

