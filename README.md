# mazeABCD_preprocessing

This pipeline preprocesses behavioral and tracking data from ABCD maze experiments. You should have done mazeABCD_maze and mazeSLEAP already! which means, you have the sleap .h5 files and the ROI .csv files within the preprocessed folder. if not, please check out those repos. 

## Files needed 
```
data/
├──raw_data/
│   ├── behaviour 
│      ├──  pinstate_files (.csv) 
│      ├── pycontrol_files (.txt)
│   └── metadata
│      └── metadata (.csv)
├──preprocessed_data/
│   ├── SLEAP/
│   │   └── sleap_tracking_files (.h5) 
│   ├── SLEAP_ROIs/ 
│   │   └── ROIs_files (.csv) 


code
├── preprocessing/ ### copy the repo to here
├── preprocessing_maze_registration/ ### see maze_registration repo 
│   ├── maze_registration.ipynb
│   ├── maze_params/ 
         ├── maze_coord_array.npy 
         ├── maze_distortion_coefficient.npy 
         ├── maze_map.npy 
         ├── maze_pixel_to_cm.npy 
         ├── maze_rotation_angle.npy 
         ├── maze_undistorted_coords.npy 
         └── maze_undistorted_coords.npy

  ```       
        


## Output Files

By going through this pipeline, you would have: 
1. `XY_raw_[mouseid]_[date]_[sess].npy`: Processed tracking coordinates
2. `Location_raw_[mouseid]_[date]_[sess].npy`: Location data
3. `trialtimes_[mouseid]_[date]_[sess].npy`: Trial timing information
4. `poke_raw_[mouseid]_[date]_[sess].npy`: Nose poke events
5. **(OPTIONAL)** `LED_raw_[mouseid]_[date]_[sess].npy`: LED stimulation data

### detailed explanation 

All of the output files would be stored in the /data/processed_data/behavioural_data/ folder. 
All of the output files would be aligned to the animal getting the first A and the last A, in 40Hz (25ms timebins) 
The sessions will be numbered according to the sessions on the day. if a session fail to pass the sync pulse match check (i.e. the number of sync pulses in pycontrol file and pinstate file does not match -- the files would not be saved out, and you might see a gap in the output files (e.g. XY_raw_mz01_20241022_0.npy, XY_raw_mz01_20241022_1.npy, XY_raw_mz01_20241022_3.npy, if the third session failed) 



1. `XY_raw_[mouseid]_[date]_[sess].npy`
     - default node: 'head_back' 
     - read from the .h5 SLEAP output, in the shape of `[time bins, 2]`, where each row is the x_, y_coord of the animals 
     - corrected from camera distortion
     - invalid transition NaNned out (default criteria: more than 2cm gap between each frame; invalid ROI transition) 

2. `Location_raw_[mouseid]_[date]_[sess].npy`: Location data

     - The corresponding location the animal is in, follows the following pattern: 

```
 1 -- 10 -- 2 -- 11 -- 3 
 |          |          |
 12         13         14
 |          |          |
 4 -- 15 -- 5 -- 16 -- 6
 |          |          |
 17        18          19
 |          |          |
 7 -- 20 -- 8 -- 21 -- 9 
```


3. `trialtimes_[mouseid]_[date]_[sess].npy`: Trial timing information

     - array of shape `[number of trials, 5]` where the columns correspond to the time animal receives `[A, B, C, D, next_A]`
     - aligned to first A (i.e. [0,0] = 0) 
   
   
4. `poke_raw_[mouseid]_[date]_[sess].npy`: Nose poke events

     - array of shape `[number of pokes, 3]` where the columns are `[poke_in_time, poke_out_time, port_num] `
   

**(OPTIONAL) -- only relevant if doing optogenetics**


5. `LED_raw_[mouseid]_[date]_[sess].npy`: LED stimulation data

    - 1D array with length the same as Location_raw and XY_raw, represent the light status of each time bin, 1 being lightON, 0 being lightOFF
    - might need to write new chunk for the function _load_led_data()  if the task you're using is not listed below:
        - ABCD_with_opto.py; ABCD_with_opto_v2.py; ABCD_with_opto_v3.py; ABCD_with_opto_v4.py; ABCD_with_opto_v5.py  







## Prerequisites

- Python 3.8+
- Required packages:
  - numpy
  - pandas
  - h5py
  - pathlib
  - logging

## Installation

1. Clone the repository to your code folder 

2. set up the conda environment for this

```
conda create --name maze_ABCD python==3.12.7
conda activate maze_ABCD
```

and then direct to your `/ceph/behrens/<yourname>/project/code` folder in terminal and run the following line to set up the environment. 

```
pip install -r mazeABCD_preprocessing/requirements.txt
```

   
## Directory Structure

```
preprocessing/
├── README.md
├── behaviour_preprocessing.ipynb # STEP4. the main behavioural preprocessing notebook, supported by preprocessing_functions.py
├── behavioural_preprocessing.log # .log file for STEP4
├── check_sleap_output.log #log for STEP1 
├── checking_sleap_output.py #STEP1 
├── get_roi_from_SLEAP.py #STEP2 
├── manual_correction.ipynb #STEP4 (OPTIONAL)
├── preprocessing_functions.py #STEP4
├── requirements.txt #prerequisite
└── transition_rate_analysis.py #STEP3
```

## Usage

### If you're running this for the first time: 

**(OPTIONAL BUT RECOMMENDED)** 
STEP1: check how accurate your sleap output has been. 

Click open the `check_sleap_output.py` file, change the `mouse_ids` and `base_folder`. In terminal, run:
  
```bash
module load miniconda
conda activate maze_ABCD
check_sleap_output.py
```

The output will be saved to `check_sleap_output.log`, which should tell you for each session the percentage of the invalid transitions, and highlight the ones not passing the threshold you've set, which you can change in the .py file by changing `percentage_nan_threshold`


STEP2: get Region_Of_Interest (ROI) data from SLEAP

- make sure you have done maze_registration and have got the maze_params folder ready. Click open the `get_roi_from_SLEAP.py`, change the `subject_to_maze_dict` to match your data. And then, run in the terminal:

```bash
conda activate maze_ABCD
python get_roi_from_SLEAP.py
```
  
- **NOTE** if you have moved the camera in the middle of the data collection, you might need to adjust the code to reflect that -- if you're running the code daily (as you do the experiments, you would just need to change the `subject_to_maze_dict` accordingly e.g. `'subject1':'maze1'` to `'subject1':'maze1_1'`
- the code reads through the `preprocessed_data/SLEAP` folder, and will identify the ones whose ROI files do not exist yet, and generate the ROI data, which are stored as .csv in `prepreprocessed_data/SLEAP_ROIs` folder, the roi data are in the shape of (frames, nodes) (i.e. each row is a frame, each column is a node (body part) registered in SLEAP).
- the code works by identifying, for a given xy coordinate, the closest point on the maze, and see which location that point corresponds to by looking at the coord_arrays.npy that you've got from maze_registration. if SLEAP outputs a NaN, it would remain NaN for ROI.


STEP3: Transition Rate analysis 

As you're doing experiment, you might need to calculate the transition rate of the animal daily to see whether you should transit them on to a new task (when they have taken the shortest path for 70 percent of the time). You should have got the ROI .csv files already. Click open the `transition_rate_analysis.py` file, put int the corresponding date, subject_id, video_sess_id and pycontrol_id. Now within your `code/mazeABCD_preprocesisng_behaviour` folder terminal, run:  

```bash
module activate maze_ABCD
python transition_rate_analysis.py
```
This will print out the transition rate for the sessions inthe terminal.

___________________________________________________________________________________________________________________________

STEP4: Main Behavioural Preprocessing 

At this stage, you have probably finished all the data collection, and have done SLEAP and SLEAP_ROIs for all the sessions. Now we're ready to generate all the .npy files that we've said we're going to do at the beginning of this README file (YAY!) To do that, you would need to work through the `behaviour_preprocessing.ipynb' on the cluster, to do that, you'd need to demand a node and open a jupyter notebook, to do that: 

within your `/code` folder (IMPORTANT!! if you're not here, might lead to import error. 

```bash

    srun --nodes=1 --ntasks-per-node=1 --cpus-per-task=8 -p gpu --gres=gpu:1 --time=12:00:00 --mem=64G --pty bash -i
        #wait for resources to be allocated
    source /etc/profile.d/modules.sh
    module load miniconda
    conda activate maze_ephys     #sometimes requires  'source activate maze_ephys'
    jupyter-notebook --no-browser --ip=0.0.0.0 --port 8888

```

And then copy the link, paste it into the "Select Kernel" entry (on the top right corner of your VSCode page). Now you could work through the notebook, change the corresponding parameters, etc. The log of running the notebook would be saved at `mazeABCD_preprocessing_behaviour/behaviour_preprocessing.log` 

- if you have certain sessions where you forgot the restart the video when restarting pycontrol or ephys, you might need to split the tracking / ROI data into two, this would be reflected in the fact that the number of sync pulses from the pinstate file would be the addition of the sync pulse counts from two sessions from ephys or pycontrol. The log for sync pulse mismatch errors would be stored at `data/preprocessed_data` folder. An example of such a mistake:

  ```
mz07-2024-10-23-164942.txt pyControl: 22 pinstate: 261
mz07-2024-10-23-165402.txt pyControl: 239 pinstate: 261
```

- if you want to rescue the sessions, go to notebook `/code/preprocessing/manual_correction.ipynb` and work through splitting the .h5 sleap files and .csv roi files, change the corresponding tracking ID in metadata




