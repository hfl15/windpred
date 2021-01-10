"""
    Run the model with the best settings.
"""


from windpred.utils.model_base import DefaultConfig
from windpred.utils import exp_dir
from windpred.utils.exp import main_spatial_duq

from windpred.baseline.duq.main import run as duqrun


def run(target, tag, eval_mode):
    features_history, features_future = [target], ['NEXT_NWP_{}'.format(target)]

    if target == 'DIR':
        tag_file_list = [None]
        for mode in ['run', 'reduce']:
            exp_dir.main(mode, eval_mode, tag, tag_file_list)
    else:
        loss = 'mve'
        layers = [32]
        csv_result_list = ['metrics_model.csv', 'metrics_nwp.csv']
        func = duqrun(features_history, features_future, loss=loss, layers=layers)
        for mode in ['run', 'reduce']:
            main_spatial_duq(target, mode, eval_mode, DefaultConfig(), tag, func, csv_result_list)





# if __name__ == '__main__':
#     # tag = tag_path(os.path.abspath(__file__), 2)
#     tag = get_tag(__file__)
#
#     target = 'V'
#     features_history, features_future = [target], ['NEXT_NWP_{}'.format(target)]
#
    # loss = 'mve'
    # layers = [32]
    # csv_result_list = ['metrics_model.csv', 'metrics_nwp.csv']
    # func = run(features_history, features_future, loss=loss, layers=layers)
    # for mode in ['run', 'reduce']:
    #     main_spatial_duq(target, mode, eval_mode, DefaultConfig(), tag, func, csv_result_list)
