from windpred.exp.base_convention import run_temporal_funcs
from windpred.expinc.base import *


if __name__ == '__main__':
    tag = get_tag(__file__)
    target = 'V'
    for key, fun in run_temporal_funcs.items():
        fun(target, '{}/{}'.format(tag, key), eval_mode)

