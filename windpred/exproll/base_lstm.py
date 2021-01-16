from windpred.exp.base_nn import run_lstm
from windpred.exproll.base import *


if __name__ == '__main__':
    tag = get_tag(__file__)
    target = 'V'
    run_lstm(target, tag, eval_mode)

