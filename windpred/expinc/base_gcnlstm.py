from windpred.exp.base_gcnlstm import run
from windpred.expinc.base import *

if __name__ == '__main__':
    tag = get_tag(__file__)
    target = 'V'
    run(target, tag, eval_mode)
