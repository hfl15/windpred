from windpred.exp.base_duq import run_grid_search
from windpred.exproll.base import *


if __name__ == '__main__':
    tag = get_tag(__file__)
    target = 'V'
    run_grid_search(target, tag, eval_mode)



