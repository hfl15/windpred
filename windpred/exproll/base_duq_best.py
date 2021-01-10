from windpred.exp.base_duq import run_best
from windpred.exproll.base import *


if __name__ == '__main__':
    tag = get_tag(__file__)
    target = 'VX'
    run_best(target, tag, eval_mode)

