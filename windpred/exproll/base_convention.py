from windpred.exp.base_convention import run
from windpred.exproll.base import *

if __name__ == '__main__':
    tag = get_tag(__file__)
    target = 'VX'
    run(target, tag, eval_mode)
