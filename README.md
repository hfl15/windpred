# MHSTN

The data and source code of the work Multi-Horizon SpatioTemporal Network (MHSTN). 

## Project structure

```
.
├── external <- Directory to store external resources, e.g., the data used in the competitor DUQ.
└── windpred
    ├── __init__.py
    ├── baseline <- There are implementations of the included competitor models.
    ├── data <- That is the constructed real-world dataset.
    │   ├── nwp <- There are nine groups of observation data (i.e., nine `.csv` files). 
    │   └── obs <- There is one group of NWP data (i.e., a `.csv` file)
    ├── eda <- There are some functions to conduct exploratory data analysis, e.g. calculation of various statistics, and visualization of data.
    ├── exp <- The basic package for conducting experiments. 
    ├── expinc <- There are experimental scripts to conduct the incremental evaluation. 
    ├── exproll <- There are experimental scripts to conduct the rolling evaluation. 
    ├── mhstn <- It is the implementation of the proposed unified framework.
    ├── utils <- There are some generic utils.
    └── requirements.txt <- It lists the required libraries.
```


## Data description

The observation and NWP data are located in the directories of `windpred/data/obs/` and `windpred/data/nwp/`, respectively. There are nine groups of observation data and one group of NWP data shared to the whole airfield.  Missing values in the observation data are represented by a number of `-9999`. The weather variables include wind speed (`V`), lateral wind speed (`VX`), longitudinal wind speed (`VY`), wind direction (`DIR`), temperature (`TP`), relative humidity (`RH`), and sea-level pressure (`SLP`). Note that the values of `VX` and `VY` in the observation data are calculated from `V` and `DIR` as they cannot be measured directly. 

For more details, please refer to the paper and the source code.


## Installation

This project is based on Python-3.6. Please apply `pip install -r requirements.txt` to install the required libraries. Please do not forget to add the project directory, i.e., `windpred`, into your Python Module Search Path. 

It should be noted that an independent (virtual) environment is needed to run the competitor DUQ because the primitive implementation is based on an earlier version of Tensorflow. Please refer to `README.md` and `requirements.txt` below `windpred/baseline/duq/` for more details.

The corresponding  can be located in 

We used whole project on Ubuntu-18.04 and macOS-10.14.6. 

## Conducting the main experiments

The directories of `expinc/` and `exproll/` correspond to the two evaluation strategies, respectively. Each of them stores the main scripts to run experiments.   

For example, the scripts located in `expinc/` are as follows. Here, a script with a prefix of `base_` corresponds to one (type) of competitor models. Our models take a prefix of `mhstn`.  

```
expinc/
├── __init__.py
├── base.py <- some common settings.
├── base_benchmark.py <- Persistence and NWP models.
├── base_convention.py <- GBRT and SVR models.
├── base_convlstm.py <- ConvLSTM(h,f,s).
├── base_convlstm_covar.py <- ConvLSTM(h,f,s,c).
├── base_convlstm_covar_all.py <- ConvLSTM(h,f,s,c*).
├── base_duq.py <- DUQ with a grid search strategy to tune hyperparameters.
├── base_duq_best.py <- DUQ(h,f).  
├── base_duq_best_covar.py <- DUQ(h,f,c).  
├── base_duq_best_covar_all.py <- DUQ(h,f,c*). 
├── base_gcnlstm.py <- GCNLSTM(h,f,s).
├── base_lstm.py <- LSTM(h), LSTM(f) and LSTM(h,f).  
├── base_lstm_spatial.py <-  LSTM(h,f,s).
├── base_mlp.py <-  MLP(h), MLP(f) and MLP(h,f).
├── mhstn.py <- MHSTN-T(h,f), MHSTN-S(h,f,s) and MHSTN-E(h,f,s).
├── mhstn_covar.py <-  MHSTN-T(h,f,c), MHSTN-S(h,f,s,c) and MHSTN-E(h,f,s,c).
├── mhstn_covar_all.py <- MHSTN-T(h,f,c*), MHSTN-S(h,f,s,c*) and MHSTN-E(h,f,s,c*).  
├── vis.py <- visualizing prediction results. 
└── vis_with_lstm_h.py <- visualizing prediction results.
```

For each of aforementioned scripts, the variable of `target`, which points to the target wind variable, can be assigned with `V`, `VX`, `VY` and `DIR`, respectively. Note that `DIR` is calculated based on `VX` and `VY`, thus the script must be operated for the latter two variables first. The experiment results are saved in the directory of `cache/`.

Please refer to the paper and source code for more details.
