from windpred.utils.base import DIR_LOG
from windpred.utils.model_base import DefaultConfig

from windpred.baseline.benchmark import main

from windpred.expinc.base import *


if __name__ == '__main__':
    tag = get_tag(__file__)

    target = 'V'
    main(target, tag, DefaultConfig(), DIR_LOG, eval_mode)









