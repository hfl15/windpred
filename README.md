# MHSTN

This is the source code for the work: A Spatiotemporal Deep Neural Network for High-Spatial-Resolution Multi-Horizon Wind Prediction.

## Project structure

- `data/`: The contributed dataset including observation and NWP data.
- `windpred/`
	- `baseline/`: There are the realizations of the included competitor models. 
	- `eda/`: There are some functions to conduct exploratory data analysis, e.g. calculation of varying correlations and statistics and visualization of data. 
	- `expinc/`: There are the experimental scripts to conduct the incremental evaluation. 
	- `exproll/`: There are the experimental scripts to conduct the rolling evaluation. 
	- `mhstn/`: There are the realizations of the proposed unified framework.
	- `utils/`: There are some general utils.

## Installation

Applying `pip install -r requirements.txt` to install the required libraries.

## Conducting the main experiments

The directories of `expinc/` and `exproll/` correspond to the two evaluation strategies, respectively. Each of them stores the main files to run the experiments.   
For example, the scripts located in `expinc/` are as follows.  
A script with a prefix of `base_` corresponds to one (type) of competitor models:
>- `base_benchmark.py`: Persistence model and NWP model
>- `base_convlstm.py`: ConvLSTM(h,f,s)
>- `base_convlstm_covar.py`: ConvLSTM(h,f,s,c)
>- `base_gcnlstm.py `: GCNLSTM(h,f,s)
>- `base_lstm.py`: LSTM(h), LSTM(f), LSTM(h,f)  
>- `base_lstm_spatial.py`: LSTM(h,f,s)
>- `base_mlp.py `: MLP(h), MLP(f), MLP(h,f)

The proposed unified model and its components correspond to the following scripts:
>- `mhstn.py`: MHSTN-T(h,f), MHSTN-S(h,f,s), MHSTN-E(h,f,s)
>- `mhstn_covar.py`: MHSTN-T(h,f,c), MHSTN-S(h,f,s,c), MHSTN-E(h,f,s,c)
>- `mhstn_covar_all.py`: MHSTN-T(h,f,c\*), MHSTN-S(h,f,s,c\*), MHSTN-E(h,f,s,c\*)  

The other two files are:
>- `vis.py `: visualizing the prediction results, 
>- `base.py`: saving common settings of experiments.

For each script to run a model, the variable of `target`, which points to the target wind variable, can be sequentially assigned with `V`, `VX`, `VY` and `DIR`. The experiment results are saved in the directory of `cache/`.

Please refer to the source code for more details.
