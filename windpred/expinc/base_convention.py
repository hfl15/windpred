from windpred.exp.base_convention import run
from windpred.expinc.base import *

if __name__ == '__main__':
    tag = tag_path(os.path.abspath(__file__), 2)
    target = 'VX'
    run(target, tag, eval_mode)

# import os
#
# from windpred.utils.base import tag_path
# from windpred.utils.model_base import DefaultConfig
# from windpred.utils import exp_dir
#
# from windpred.baseline.convention import main, MODELS
#
# from windpred.expinc.base import eval_mode
#
#
# if __name__ == '__main__':
#     tag = tag_path(os.path.abspath(__file__), 2)
#     target = 'VX'
#
#     if target == 'DIR':
#         for model_name in MODELS.keys():
#             file_in = os.path.join(tag, model_name)
#             tag_file_list = [None]
#             exp_dir.main('run', eval_mode, file_in, tag_file_list)
#             exp_dir.main('reduce', eval_mode, file_in, tag_file_list)
#     else:
#         for model_name in MODELS.keys():
#             main(tag, DefaultConfig(), target, 'run', eval_mode, model_name)
#             main(tag, DefaultConfig(), target, 'reduce', eval_mode, model_name)


