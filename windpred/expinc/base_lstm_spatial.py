from windpred.exp.base_lstm_spatial import run
from windpred.expinc.base import *


if __name__ == '__main__':
    tag = get_tag(__file__)
    target = 'V'
    run(target, tag, eval_mode)
