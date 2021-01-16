from windpred.exp.base_nn import run_mlp
from windpred.expinc.base import *

if __name__ == '__main__':
    tag = tag_path(os.path.abspath(__file__), 2)
    target = 'V'
    run_mlp(target, tag, eval_mode)

