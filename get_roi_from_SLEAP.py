### this script is for getting the ROI from the SLEAP output 

import h5py
import numpy as np
import pandas as pd
import scipy.spatial 
import os

## variables to change
subject_to_maze_dict = {'mz05': 'maze1', 'mz06': 'maze2', 'mz07': 'maze1', 'mz09': 'maze2', 'mz10': 'maze2'}



### if you have moved the camera, the available maze might be something like maze2_2, maze2_3 etc. then maybe you need to change the subject_to_maze_dict to reflect that, or load the metadata accordingly. but then: CHANGE THE SCRIPT! 

available_maze = ['maze1', 'maze2', 'maze2_2']
base_folder = "/ceph/behrens/mingyu_zhu/vHPC_mPFC"



### the rest should be automatically done 
sleap_folder = f"{base_folder}/data/preprocessed_data/SLEAP"
roi_folder = f"{base_folder}/data/preprocessed_data/SLEAP_ROIs"
if not os.path.exists(roi_folder):
    os.makedirs(roi_folder)

behaviour_folder = f"{base_folder}/data/raw_data/behaviour"

maze_params_folder = f"{base_folder}/code/preprocessing_maze_registration/maze_params"

if not os.path.exists(maze_params_folder):
    raise ValueError(f"Maze params folder {maze_params_folder} does not exist")
## loading maze map 
coord_array_dict = dict()
maze_map_dict = dict()
for maze in available_maze:
    maze_map_dict[maze] =np.load(f"{maze_params_folder}/{maze}_map.npy")
    coord_array_dict[maze] = np.load(f"{maze_params_folder}/{maze}_coord_array.npy")

## essential functions: 
def get_closest_node(cdktree, curr_coord_x, curr_coord_y):
    nearest_node_ind = cdktree.data[cdktree.query([curr_coord_x,curr_coord_y])[1]].astype(int)
    return nearest_node_ind


for file in os.listdir(sleap_folder):
    if file == '.DS_Store':
        continue
    #print(file)
    curr_session_id = file.split('.')[0].split('_')[-1]
    subject_id = file.split('_')[0]
    output_filename = f"{subject_id}_{curr_session_id}.csv"
    if (output_filename in os.listdir(roi_folder))==True:
        print(f"{subject_id}-{curr_session_id}, file already exists") ## skip if ROI file already exists 
        continue
    date_string_as_int = int("".join(curr_session_id.split('-')[:3]))
    maze_curr = subject_to_maze_dict[subject_id]

    data_path_f = f"{sleap_folder}/{file}"
    data = h5py.File(data_path_f,'r')
    data_track = np.round(data['tracks'][0]).astype(int)
    data_dict = dict()
    bp_array = np.ndarray(shape = ((len(data['node_names']), len(data_track[0][0]))))
    majority_array = np.ndarray(shape = ((len(data_track[0][0]))))
    cdktree_coord = scipy.spatial.cKDTree(coord_array_dict[maze_curr])
    bp_array[:] = np.nan
    x_limit = len(maze_map_dict[maze_curr])
    y_limit = len(maze_map_dict[maze_curr][0])
    for j in range(len(data_track[0][0])):
        for i in range(len(data['node_names'])):
            if data_track[0][i][j] == 0 or data_track[0][i][j] >= x_limit or data_track[1][i][j] >= y_limit:
                continue
            else:
                closest_node = get_closest_node(cdktree_coord,data_track[1,i,j], data_track[0,i,j]) ## the indexing here reverses the x-y position to map the trajectory onto the maze_map 
                #print(closest_node)
                bp_array[i,j] = maze_map_dict[maze_curr][closest_node[0], closest_node[1]]
            majority_array[j] = max(set(bp_array[:,j]), key = list(bp_array[:,j]).count)
            #print(bp_array[:,j],majority_array[j])
                #node_pos.append(maze2_map[data_track[0,i,j], data_track[1,i,j]])
            #print(node_pos)
    for i in range(len(data['node_names'])):
        data_dict[str(data['node_names'][i])[2:-1]] = bp_array[i]
    data_dict['majority'] = majority_array
    data_framed = pd.DataFrame(data_dict)
    data_framed.to_csv(f"{roi_folder}/{output_filename}")
    print(output_filename + str(' -- Done!'))

