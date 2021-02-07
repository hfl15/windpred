import os

from windpred.utils.base import DIR_LOG
from windpred.utils.base import tag_path
from windpred.utils.model_base import DefaultConfig

from windpred.baseline.benchmark import main

from windpred.exproll.base import eval_mode


if __name__ == '__main__':
    tag = tag_path(os.path.abspath(__file__), 2)

    target = 'V'
    main(tag, DefaultConfig(), DIR_LOG, eval_mode)









