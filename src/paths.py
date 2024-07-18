import os 

### Chnage this to your own path ###
current_dir = os.getcwd()
path = os.path.join(current_dir, '..', 'data')
param = os.path.join(path, 'Formation_2022-Parameter.csv') 
diff = os.path.join(path, 'delta_features_041524.csv')
life = os.path.join(path,'one_time_features_041524.csv')
rpt = os.path.join(path,'rpt_summary_041524.csv')
form = os.path.join(path,'formation_cycle_info_042124.csv')
electrode = os.path.join(path,'electrode_info_04152024.csv')