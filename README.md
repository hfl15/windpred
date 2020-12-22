# MHSTN

The data and source code of the work: A Spatiotemporal Deep Neural Network for High-Spatial-Resolution Multi-Horizon Wind Prediction.

## Project structure

- `data/`:
	- `obs/`: The nine groups of observation data (corresponding to nine `.csv` files). 
	- `nwp/`: The one group of NWP data (corresponding to a `.csv` file).
- `windpred/`
	- `baseline/`: The realizations of the included competitor models. 
	- `eda/`: Some functions to conduct exploratory data analysis, e.g. calculation of varying correlations and statistics and visualization of data. 
	- `expinc/`: The experimental scripts to conduct the incremental evaluation. 
	- `exproll/`: The experimental scripts to conduct the rolling evaluation. 
	- `mhstn/`: The realizations of the proposed unified framework.
	- `utils/`: Some general utils.

## Data description

The observation and NWP data are located in the directories of `data/obs/` and `data/nwp/`, respectively. There are nine groups of observation data and one group of NWP data shared to the whole airfield.  Missing values in the observation data are represented by a number of `-9999`. The weather variables include wind speed (`V`), lateral wind speed (`VX`), longitudinal wind speed (`VY`), wind direction (`DIR`), temperature (`TP`), relative humidity (`RH`), and sea-level pressure (`SLP`). Note that the values of `VX` and `VY` in the observation data are calculated from `V` and `DIR` as they cannot be measured directly. 

For more details, please refer to the paper and the source code.

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
>- `base.py`: saving the common settings of experiments.

For each of aforementioned scripts, the variable of `target`, which points to the target wind variable, can be assigned with `V`, `VX`, `VY` and `DIR`, respectively. Note that `DIR` is calculated based on `VX` and `VY`, thus the script must be operated for the latter two variables first. The experiment results are saved in the directory of `cache/`.

Please refer to the source code for more details.
