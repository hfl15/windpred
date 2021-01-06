import os

from windpred.utils.base import tag_path
from windpred.utils.model_base import DefaultConfig

from windpred.baseline.convention import main, MODELS

from windpred.expinc.base import eval_mode


if __name__ == '__main__':
    tag = tag_path(os.path.abspath(__file__), 2)
    mode = 'run'
    target = 'V'
    for model_name in MODELS.keys():
        main(tag, DefaultConfig(), target, mode, eval_mode, model_name)

