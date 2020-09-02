import os

from windpred.utils.base import tag_path

from windpred.utils.model_base import DefaultConfig, BaseLSTM
from windpred.utils import exp_dir
from windpred.baseline import temporal_nn
from windpred.expslid.base import eval_mode


if __name__ == '__main__':
    tag = tag_path(os.path.abspath(__file__), 2)

    target = 'DIR10'
    mode_list = ['run-history', 'run-future', 'run-history_future']

    if target == 'DIR10':
        for mode in mode_list:
            tag_file_list = [None]
            exp_dir.main('run', eval_mode, tag, tag_file_list)
            exp_dir.main('reduce', eval_mode, tag, tag_file_list)
    else:
        for mode in mode_list:
            temporal_nn.main(tag, DefaultConfig, target, mode, eval_mode, BaseLSTM)







