import os

from windpred.utils.base import tag_path
from windpred.utils.model_base import DefaultConfig

from windpred.baseline import gcnlstm
from windpred.utils import exp_dir

from windpred.exp.base_gcnlstm import run
from windpred.exproll.base import *


if __name__ == '__main__':
    tag = get_tag(__file__)
    target = 'V'
    run(target, tag, eval_mode)
    # model_name = 'gcn_seq_lstm_seq'
    #
    # if target == 'DIR':
    #     tag_file_list = [model_name]
    #     exp_dir.main_old('run', eval_mode, tag, tag_file_list)
    #     exp_dir.main_old('reduce', eval_mode, tag, tag_file_list)
    # else:
    #     features_history, features_future = [target], ['NEXT_NWP_{}'.format(target)]
    #     for mode in ['run', 'reduce']:
    #         adjacency_norm = 'localpooling_filter'
    #         gcnlstm.main(target, mode, eval_mode, DefaultConfig, tag, model_name, features_history, features_future, adjacency_norm)

