
### sometimes sleap doesn't work as well -- this file is for checking if the output of sleap has a lot of NaN values, and if so, remove the file, so that you can rerun it. 



import glob
import h5py
import numpy as np
import os
import logging
import pandas as pd
current_dir = os.path.dirname(os.path.abspath(__file__))

# Create the full path for the log file
log_file_path = os.path.join(current_dir, 'check_sleap_output.log')

# Configure the logger
logging.basicConfig(
    filename=log_file_path,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Example log entry
logging.info("Checking sleap output.")


## things to change here 
base_folder = '/ceph/behrens/mingyu_zhu/vHPC_mPFC'
mouse_ids = ['mz05','mz06','mz07','mz09','mz10']

## from here on, it should be the same 
sleap_folder = f"{base_folder}/data/preprocessed_data/SLEAP"

available_tracking = glob.glob(f"{sleap_folder}/*.h5*")  
available_tracking = sorted(available_tracking)
percentage_nan_threshold = 0.05
not_passed_files = 0 
nan_percentage_list = []
target_node = 'head_back'
data_test = h5py.File(available_tracking[0], 'r')

for i in range(len(data_test['node_names'])):
    if data_test['node_names'][i].astype(str) != target_node:
        continue
    else:
        node_ind = i 

for f in available_tracking:
    data_test = h5py.File(f, 'r')
    data_track = np.round(data_test['tracks'][0][:, node_ind, :],decimals = 3).T
    nan_percentage_curr = len(np.where(np.isnan(data_track)[:,0] == True)[0])/len(data_track)
    nan_percentage_list.append(nan_percentage_curr)
    
    if nan_percentage_curr > percentage_nan_threshold:
        logging.info(f"{f.split('/')[-1]} - NaN %: {nan_percentage_curr:.4f} -- not passing threshold")
        not_passed_files += 1
        continue
    logging.info(f"{f.split('/')[-1]} - NaN %: {nan_percentage_curr:.4f}")


logging.info(f"Not passed files percentage: {(not_passed_files/len(available_tracking))*100:.2f}%")

    
