from windpred.utils.model_base import DefaultConfig

from windpred.baseline import gcnlstm
from windpred.utils import exp_dir


def run(target, tag, eval_mode):
    model_name = 'gcn_seq_lstm_seq'

    if target == 'DIR':
        tag_file_list = [model_name]
        for mode in ['run', 'reduce']:
            exp_dir.main(mode, DefaultConfig(), eval_mode, tag, tag_file_list)
    else:
        features_history, features_future = [target], ['NEXT_NWP_{}'.format(target)]
        adjacency_norm = 'localpooling_filter'
        for mode in ['run', 'reduce']:
            gcnlstm.main(target, mode, eval_mode, DefaultConfig, tag, model_name, features_history, features_future,
                         adjacency_norm)

