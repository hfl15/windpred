# MHSTN

This is the source code for the work: A Spatiotemporal Deep Neural Network for High-Spatial-Resolution Multi-Horizon Wind Prediction.

## Project structure

- `data/`: The contributed dataset including observation and NWP data.
- `windpred/`
	- `baseline/`: The realizations of the included competitor models. 
	- `eda/`: Some functions to conduct exploratory data analysis, e.g. calculating varying correlations and statistics, and visualizing of data.
	- `expinc/`: The scripts to run experiments of the incremental evaluation. 
	- `exproll/`: The scripts to run experiments of the rolling evaluation. 
	- `mhstn/`: The realization of the proposed unified framework.
	- `utils/`: Some general utils.

## Installation

Applying `pip install -r requirements.txt` to install the required libraries.

## Conducting the main experiments

There are two evaluation strategies corresponding to the directories of `expinc/` and `exproll/`, respectively. Each of them stores the main files to run the experiments.   
For example `expinc/`, the scripts are as follows.  
A script with a prefix of `base_` corresponds one (type) of competitor models:
>- `base_convlstm.py`: ConvLSTM(h,f,s)
>- `base_convlstm_covar.py`: ConvLSTM(h,f,s,c)
>- `base_gcnlstm.py `: GCNLSTM(h,f,s)
>- `base_lstm_spatial.py`: LSTM(h,f,s)
>- `base_mlp.py `: MLP(h), MLP(f), MLP(h,f)
>- `base_lstm.py`: LSTM(h), LSTM(f), LSTM(h,f)  

The proposed unified model and its components correspond to following scripts:
>- `mhstn.py`: MHSTN-T(h,f), MHSTN-S(h,f,s), MHSTN-E(h,f,s)
>- `mhstn_covar.py`: MHSTN-T(h,f,c), MHSTN-S(h,f,s,c), MHSTN-E(h,f,s,c)
>- `mhstn_covar_all.py`: MHSTN-T(h,f,c\*), MHSTN-S(h,f,s,c\*), MHSTN-E(h,f,s,c\*)  

The other two files:
>- `vis.py `: visualizing the prediction results.
>- `base.py`: saving common settings of experiments.


For each script to run a model, the variable of `target`, which points to the target wind variable, can be sequentially assigned with `V`, `VX`, `VY` and `DIR`. The experiment results will be saved in the directory of `cache/`.

Please refer to the source code for more details.