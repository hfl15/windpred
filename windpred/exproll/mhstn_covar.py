from windpred.exp.mhstn import run_covar
from windpred.exproll.base import *

if __name__ == '__main__':
    tag = get_tag(__file__)
    target = 'V'
    run_covar(target, tag, eval_mode)
