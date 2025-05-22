import numpy as np
import pandas as pd 
import json
from pathlib import Path
import matplotlib.pyplot as plt


## change here to where you have your behavioural data
base_folder =  "/ceph/behrens/mingyu_zhu/vHPC_mPFC/data" 
behaviour_folder = f"{base_folder}/raw_data/behaviour"
roi_folder = f"{base_folder}/preprocessed_data/SLEAP_ROIs"



###things to change here: 
date = '2024-12-02'

subject_id = 'mz09'
video_sess_ids = [111046,120815,132527,143356]
pycontrol_ids = [111111,120838,132552,143415]


## things that probably don't need to change: 
node_used = 'head_back' 

sampling_frequency = 1000/60 #the duration of the time bin we want, which will be used to convert the pycontrol timestamp (in ms)
## since the frame rate is 60 hz, we want the time bin to be in 60 Hz as well, which means each time bin is 1000/60 


## the rest should be the same: 

pinstate_session = []
tracking_session = []
pycontrol_session = []
for sess in range(len(video_sess_ids)):

    pinstate_session.append(subject_id+'_pinstate_'+date+'-'+str(video_sess_ids[sess]))
    tracking_session.append(subject_id+'_'+date+'-'+str(video_sess_ids[sess]))
    pycontrol_session.append(subject_id+'-'+date+'-'+str(pycontrol_ids[sess]))

    
maze_grid = np.arange(1,10).reshape(3,3)

reward_state = ['A_on', 'B_on','C_on', 'D_on']


### function to calculate the shortest path between two nodes in the maze
def cal_dist(x,y):
    dist = (abs(np.where(maze_grid == x)[0] - np.where(maze_grid == y)[0]) + abs(np.where(maze_grid == x)[1] - np.where(maze_grid == y)[1]))[0]
    return dist



all_pycontrol_files = []
all_pinstate_files = []
all_ROI_files = []
for i in range(len(pinstate_session)):
    all_pinstate_files.append(behaviour_folder+'/'+pinstate_session[i]+'.csv')
    all_ROI_files.append(roi_folder+'/'+tracking_session[i]+'.csv')
    all_pycontrol_files.append(behaviour_folder+'/'+pycontrol_session[i]+'.txt')



for i in range(len(all_pycontrol_files)):
    data_pycontrol = open(all_pycontrol_files[i], 'r')
    print(all_pycontrol_files[i].split('/')[-1])
    pycontrol_dict = dict()
    for line in data_pycontrol:
        if line[0] == 'V' and line.split(' ')[2] =='active_poke':
            #print(line)
            task_str = (line.split(' ')[-1])[1:-2].split(',')
            
            task_list = list()
            for j in range(len(task_str)):
            #    print(int(task_str[j]))
                task_list.append(int(task_str[j]))
            pycontrol_dict['Task'] = np.array(task_list).tolist()

        if line[0:11] == 'I Task name':
            task_name = (line.split(':')[1][1:-1])
            pycontrol_dict['Task_name'] = task_name
        
        elif line.startswith('S '):  # State definitions
            state_dict = json.loads(line[2:])
        elif line.startswith('E '):  # Event definitions
            events_dict = json.loads(line[2:])
            #with open(f"{preprocessing_params_folder}/event_reference_folder/{task_name}_event_reference.json", 'r') as fp:
            #    event_dict = json.load(fp)
    
    event_dict = events_dict | state_dict
    data_pycontrol = open(all_pycontrol_files[i], 'r')
    event_list = [0,0]
    total_events_count = 0
    for line in data_pycontrol:
        if line[0] != 'D':
            continue
        total_events_count+=1
        if np.sum(event_list)==0:
            event_list[0] = (int(line.split(' ')[1])/sampling_frequency)
            event_list[1] = (int(line.split(' ')[-1][:-1]))
            event_array = np.array(event_list)
        else:
            event_list[0] = (int(line.split(' ')[1])/sampling_frequency)
            event_list[1] = (int(line.split(' ')[-1][:-1]))
            event_array = np.vstack((event_array, np.array(event_list)))
    timestamps = event_array.T[0]
    for key in event_dict.keys():
        ref_num = event_dict[key]
        #print(ref_num)
        event_time_ind = np.where(event_array[:,1] == ref_num)[0]
        event_time = np.array([timestamps[j] for j in event_time_ind]).tolist()
        #print(event_time)
        pycontrol_dict[key] = event_time
    
    
    for a, state in enumerate(reward_state):
        if a == 0:
            trial_time_array = np.ndarray(shape = (len(pycontrol_dict[state]), len(reward_state)))
            trial_time_array[:] = np.nan
            trial_time_array[:,a] = np.array(pycontrol_dict[state])
        else:
            trial_time_array[:len(pycontrol_dict[state]), a] = np.array(pycontrol_dict[state])
    
    shortest_dist_list = []
    for loc in range(len(pycontrol_dict['Task'])):
        try:
            shortest_dist_list.append(cal_dist(pycontrol_dict['Task'][loc],pycontrol_dict['Task'][loc+1]))
        except:
            shortest_dist_list.append(cal_dist(pycontrol_dict['Task'][loc], pycontrol_dict['Task'][0]))
    print('Task is: ', pycontrol_dict['Task'], 'shortest path would be: ', shortest_dist_list)
    

    pinstate_file = np.loadtxt(all_pinstate_files[i])
    
    pinstate_time_ids = np.unique(pinstate_file)
    instances = (np.where(pinstate_file == pinstate_time_ids[1])[0])
    adjusted_trial_time = trial_time_array + instances[0]
    
    print('checking, the number detected in pinstate: ', len(instances)/3, 'number of rsync from pycontrol files: ', len(pycontrol_dict['rsync']))
    print('Num of trials = ', len(adjusted_trial_time))
    #print(adjusted_trial_time)
    tracking_data = pd.read_csv(all_ROI_files[i])
    
    for sta in range(len(reward_state)):
        state_loc = []
        for tr in range(len(adjusted_trial_time)):
            try:
                state_loc.append(tracking_data[node_used].iloc[int(adjusted_trial_time[tr][sta])])
            except:
                continue ## to skip NaN 

        print('checking for state ', sta, ' that animal has been in location: ', np.unique(state_loc))
        plt.hist(state_loc,np.arange(1,21,1))
    
    transition_list = []
    correct = 0
    transition_list_template = np.tile(shortest_dist_list, len(adjusted_trial_time))
    for re in range(len(adjusted_trial_time.flatten())-1):

        if np.isnan(adjusted_trial_time.flatten()[re+1]) == True:
            break
        curr_re_frame = int(adjusted_trial_time.flatten()[re])
        next_re_frame = int(adjusted_trial_time.flatten()[re+1])
        trajectory = np.unique(tracking_data[node_used].iloc[curr_re_frame:next_re_frame])
        node_passed = len(np.where(trajectory < 10)[0]) -1 
        transition_list.append(node_passed)
        if node_passed == transition_list_template[re]:
            correct+=1
    print('transition rate = ', correct/len(transition_list))


        
        
    

        
        
        