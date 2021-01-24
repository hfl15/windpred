from windpred.exp.base_convention import run_local_funcs
from windpred.exproll.base import *


if __name__ == '__main__':
    tag = get_tag(__file__)
    target = 'V'
    for key, fun in run_local_funcs.items():
        fun(target, '{}/{}'.format(tag, key), eval_mode)

