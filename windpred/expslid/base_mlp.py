import os

from windpred.utils.base import tag_path

from windpred.utils.model_base import DefaultConfig, BaseMLP
from windpred.baseline.standard_nn import main
from windpred.expslid.base import eval_mode


if __name__ == '__main__':
    tag = tag_path(os.path.abspath(__file__), 2)

    target = 'SPD10'
    mode_list = ['run-history', 'run-future', 'run-history_future']
    for mode in mode_list:
        main(tag, DefaultConfig, target, mode, eval_mode, BaseMLP)







