from windpred.exp.mhstn import run
from windpred.exproll.base import *


if __name__ == '__main__':
    tag = get_tag(__file__)
    target = 'V'
    run(target, tag, eval_mode)


