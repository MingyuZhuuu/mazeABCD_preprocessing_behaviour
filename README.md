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
conda create --name <env_name> python==3.12.7
conda activate <env_name>
```

and then direct to your `/ceph/behrens/<yourname>/project/code` folder in terminal and run the following line to set up the environment. 

```
pip install -r mazeABCD_preprocessing/requirements.txt
```

   
## Directory Structure

```
preprocessing/
├── README.md
├── 
├── 
├── 
├── 
├── 
└── 
```

## Usage

### If you're running this for the first time: 

**(OPTIONAL BUT RECOMMENDED)** 
STEP1: check how accurate your sleap output has been 




STEP2: running the script 
- Direct to your `/code` folder in the terminal (important, or else might lead to import error).
- Activate the corresponding conda environment: 
```bash
conda activate maze_ABCD 
```

Run the preprocessing pipeline:

```bash

```

STEP3: check for files with sync pulse errors and manually correct for them 
- the log for sync pulse mismatch errors would be stored at 
  ```
- go to notebook `/code/preprocessing/manual_correction.ipynb` and work through splitting the .h5 sleap files and .csv roi files, change the corresponding tracking ID in metadata **(IMPORTANT)**
- rerun
```bash

```



### Optional Command Line Arguments

- `--mouse-ids`: List of mouse IDs to process (default: all mice)
- `--metadata-folder`: Path to metadata folder
- `--debug`: Enable debug logging. all the outputs within the terminal will also be written to the .log file 

- e.g. python run_preprocessing.py --mouse-ids mz05 mz06 --debug


## 




## Folder Structure 

```
├── code/ 
    │   ├──ABCD_preprocessing/
    │   ├──ABCD_preprocessing_ephys/
    │   ├──mazeSLEAP/

├── data/ 
    │   ├──raw_data/
    │   ├──preprocessed_data/
    │   ├──processed_data/
    │   ├──analysis_data/
    │   ├──experiment_info.json

```
### raw_data

```

├── raw_data/
    │   ├── ephys/
    │   ├── behaviour/  
    │   ├── metadata/

```

### preprocessed_data

```

├── preprocessed_data/
    │   ├── SLEAP/
    │   ├── SLEAP_ROIs/
    │   ├── video/
    │   ├── LFP/
    │   ├── kilosort/
    │   └── Sleap/

```

### processed_data

```

├── processed_data/
    │   ├── neuron_raw/
    │   ├── behaviour_raw/
    │   ├── trialtimes_raw/
    │   ├── task_raw/
    │   ├── LED_raw/  ## if optogenetics
    │   └── poketime_raw/

```







