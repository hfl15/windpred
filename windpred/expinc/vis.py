from windpred.utils.model_base import DefaultConfig
from windpred.exp.vis import plot, plot_dir
from windpred.expinc.base import *


if __name__ == '__main__':
    tag = get_tag(__file__)

    plot(tag, DefaultConfig(), 'V', 'expinc_mhstn_covar')
    plot(tag, DefaultConfig(), 'VX', 'expinc_mhstn')
    plot(tag, DefaultConfig(), 'VY', 'expinc_mhstn')
    plot_dir(tag, DefaultConfig(), 'DIR', 'expinc_mhstn_covar')















