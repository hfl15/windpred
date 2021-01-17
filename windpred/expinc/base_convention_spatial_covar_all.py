from windpred.exp.base_convention import run_spatial_covar
from windpred.expinc.base import *

if __name__ == '__main__':
    tag = get_tag(__file__)
    target = 'V'
    run_spatial_covar(target, tag, eval_mode)

