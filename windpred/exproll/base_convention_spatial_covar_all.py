from windpred.exp.base_convention import run_spatial_covar_all
from windpred.exproll.base import *

if __name__ == '__main__':
    tag = get_tag(__file__)
    target = 'V'
    run_spatial_covar_all(target, tag, eval_mode)

