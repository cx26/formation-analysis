# ***Data-Driven Analysis of Battery Formation*** 

This repository contains half cell measurement data and code (for plotting, simulation and experiment design) accompanying the paper *Data-Driven Analysis of Battery Formation Reveals the Role of Electrode Utilization in Extending Cycle Life* by Xiao Cui et al.(2024)  

- **[data](data)**: 
    - extracted summary information from [aging cycles, diagnostic cycles](data/rpt_summary_041524.csv) and [formation cycles](data/formation_cycle_info_042124.csv); [resistances](data/delta_features_041524.csv); [electrode-specifics](data/electrode_info_04152024.csv) fitted using differential voltage analysis; [formation parameters](data/Formation_2022-Parameter.csv) for each cell; [cycle life](data/one_time_features_041524.csv).
    - dvf_data contains the half cell measurement data 
- **[src](src)** includes code to:
    - load in the necessary dataframes with cells that successfully reached end of life ([load_data.py](src/load_data.py))
    - extract the cycling summary information for each cell from the TRI BEEP processed cycling data ([nova_feature_helper_functions_commented.py](src/nova_feature_helper_functions_commented.py))
    - define the file paths ([paths.py](src/paths.py))
    - fit the full cell VQ curve with fresh half cell curve using differential voltage analysis ([DVF_functions.py](src/DVF_functions.py))
- **[notebooks](notebooks)** contains notebooks to:
    - generate the plots in the paper [Figure 1](notebooks/FP1_fig_1.ipynb), [Figure 2-3](notebooks/FP1_fig_2.ipynb), [Figure 4](notebooks/FP1_fig_4.ipynb), and [Figure 6](notebooks/FP1_fig_6.ipynb)
    - demonstration of how to simulate full cell curve from half cell data  ([halfcell_fitting_updated_2024_cleaned.ipynb](notebooks/halfcell_fitting_updated_2024_cleaned.ipynb))
    - prepare the half cell data for differential voltage analysis([PE_NE_curve_prep.ipynb](notebooks/PE_NE_curve_prep.ipynb))

**Additional notes**: Please update the file path to run the code. The cycle summary extraction code is written based on the cycling data processed by TRI BEEP. 
