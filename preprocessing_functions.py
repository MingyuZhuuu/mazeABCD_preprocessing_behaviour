import h5py
import numpy as np
import os 
import pandas as pd 
import json
from pathlib import Path
import scipy.spatial 
import matplotlib.pyplot as plt
import math

def cal_dist(xy1, xy2):
    return math.sqrt((xy2[0] - xy1[0])**2 + (xy2[1] - xy1[1])**2)


def calculate_rotation_angle(points_array): ## calculate the average rotational angle 
    # Assuming points[3] (mid-left) and points[5] (mid-right) are horizontally aligned and points[1] and points[7] are vertically aligned
    p1 = points_array[3]
    p2 = points_array[5]
    p3 = points_array[1]
    p4 = points_array[7]


    delta_x = p2[0] - p1[0]
    delta_y = p2[1] - p1[1]
    delta_x_v = p4[0] - p3[0]
    delta_y_v = p4[1] - p3[1]

    # Calculate angle in radians and convert to degrees
    angle_rad = np.arctan2(delta_y, delta_x)
    angle_deg = np.degrees(angle_rad)
    angle_rad_v = np.arctan2(delta_y_v, delta_x_v)
    angle_deg_v = (np.degrees(angle_rad_v) - 90)
    angle_deg_avg = (angle_deg+angle_deg_v)/2
    print(f'Rotation Angle: {angle_deg:.2f} degrees for horizontal, and {angle_deg_v:.2f} degrees for vertical. Average: {angle_deg_avg:.2f}')
    return angle_deg_avg


def calculate_pixel_to_cum(edge_coords,edge_cm = 11):
    edge_diff = []
    edge_diff.append(cal_dist(edge_coords[0],edge_coords[5]))
    edge_diff.append(cal_dist(edge_coords[1],edge_coords[4]))
    edge_diff.append(cal_dist(edge_coords[2],edge_coords[7]))
    edge_diff.append(cal_dist(edge_coords[3],edge_coords[6]))
    avg_pixel_to_cm = (np.mean(edge_diff))/edge_cm
    return avg_pixel_to_cm



def calculate_correct_point(central_node, angle_degrees, pixel_to_cm, node_to_node = 18):
    # Convert angle from degrees to radians
    angle_radians = math.radians(angle_degrees)
    x1 = central_node[0]
    y1 = central_node[1]
    # Calculate new coordinates
    x2 = x1 + node_to_node*pixel_to_cm * math.cos(angle_radians)
    y2 = y1 + node_to_node*pixel_to_cm * math.sin(angle_radians)
    new_coords = np.array([x2,y2])
    return new_coords

def construct_correct_maze_coords(points_array,angle_degree, pixel_to_cm, num_node = 9):
    correct_maze_array = np.ndarray((num_node, 2))
    central_to_corner_dist = math.sqrt(18**2+18**2)
    correct_maze_array[0] = calculate_correct_point(points_array[4], 225+angle_degree, pixel_to_cm, node_to_node=central_to_corner_dist)
    correct_maze_array[1] = calculate_correct_point(points_array[4], 270+angle_degree, pixel_to_cm)
    correct_maze_array[2] = calculate_correct_point(points_array[4], 315+angle_degree, pixel_to_cm, node_to_node=central_to_corner_dist)
    correct_maze_array[3] = calculate_correct_point(points_array[4], 180+angle_degree, pixel_to_cm)
    correct_maze_array[4] = points_array[4]
    correct_maze_array[5] = calculate_correct_point(points_array[4], angle_degree, pixel_to_cm)
    correct_maze_array[6] = calculate_correct_point(points_array[4], 135+angle_degree, pixel_to_cm, node_to_node=central_to_corner_dist)
    correct_maze_array[7] = calculate_correct_point(points_array[4], 90+ angle_degree, pixel_to_cm)
    correct_maze_array[8] = calculate_correct_point(points_array[4], 45+angle_degree, pixel_to_cm, node_to_node=central_to_corner_dist)
    return correct_maze_array



from scipy.optimize import least_squares

# Distortion model
def distort_points(params, undistorted_points):
    k1, k2, k3, p1, p2 = params
    x = undistorted_points[:, 0]
    y = undistorted_points[:, 1]
    r2 = x**2 + y**2

    # Radial distortion
    radial = 1 + k1 * r2 + k2 * r2**2 + k3 * r2**3

    # Tangential distortion
    x_tangential = 2 * p1 * x * y + p2 * (r2 + 2 * x**2)
    y_tangential = p1 * (r2 + 2 * y**2) + 2 * p2 * x * y

    x_distorted = x * radial + x_tangential
    y_distorted = y * radial + y_tangential

    return np.vstack((x_distorted, y_distorted)).T

# Reprojection error function
def reprojection_error(params, undistorted_points, observed_distorted_points):
    predicted = distort_points(params, undistorted_points)
    error = (predicted - observed_distorted_points).ravel()  # Flatten for least_squares
    return error





# Distortion model (forward mapping)
def distort_points(params, undistorted_points):
    k1, k2, k3, p1, p2 = params
    x = undistorted_points[:, 0]
    y = undistorted_points[:, 1]
    r2 = x**2 + y**2

    radial = 1 + k1 * r2 + k2 * r2**2 + k3 * r2**3
    x_tangential = 2 * p1 * x * y + p2 * (r2 + 2 * x**2)
    y_tangential = p1 * (r2 + 2 * y**2) + 2 * p2 * x * y

    x_distorted = x * radial + x_tangential
    y_distorted = y * radial + y_tangential

    return np.vstack((x_distorted, y_distorted)).T

# Undistortion (inverse mapping using iterative refinement)
def undistort_points(tracking_data, params, max_iter=10):
    undistorted_points = tracking_data.copy()  # Initial guess

    for _ in range(max_iter):
        distorted_guess = distort_points(params, undistorted_points)
        delta = tracking_data - distorted_guess
        undistorted_points += delta  # Adjust based on the error

    return undistorted_points



valid_connections = {
    1: [10, 12],  # Node 1 connects to Bridge 10 (1-2) and Bridge 12 (1-4)
    2: [10, 11, 13, 15],  # Node 2 connects to Bridge 10 (1-2), Bridge 11 (2-3), Bridge 13 (2-5), Bridge 15 (4-5)
    3: [11, 14],  # Node 3 connects to Bridge 11 (2-3) and Bridge 14 (3-6)
    4: [12, 13, 15, 17],  # Node 4 connects to Bridge 12 (1-4), Bridge 13 (2-5), Bridge 15 (4-5), Bridge 17 (4-7)
    5: [13, 15, 16, 18],  # Node 5 connects to Bridge 13 (2-5), Bridge 15 (4-5), Bridge 16 (5-6), Bridge 18 (5-8)
    6: [14, 16, 19],  # Node 6 connects to Bridge 14 (3-6), Bridge 16 (5-6), Bridge 19 (6-9)
    7: [17, 20],  # Node 7 connects to Bridge 17 (4-7) and Bridge 20 (7-8)
    8: [18, 20, 21],  # Node 8 connects to Bridge 18 (5-8), Bridge 20 (7-8), Bridge 21 (8-9)
    9: [19, 21],   # Node 9 connects to Bridge 19 (6-9) and Bridge 21 (8-9)
    10: [1, 2],  # Bridge 10 connects Node 1 and Node 2
    11: [2, 3],  # Bridge 11 connects Node 2 and Node 3
    12: [1, 4],  # Bridge 12 connects Node 1 and Node 4
    13: [2, 5],  # Bridge 13 connects Node 2 and Node 5
    14: [3, 6],  # Bridge 14 connects Node 3 and Node 6
    15: [4, 5],  # Bridge 15 connects Node 4 and Node 5
    16: [5, 6],  # Bridge 16 connects Node 5 and Node 6
    17: [4, 7],  # Bridge 17 connects Node 4 and Node 7
    18: [5, 8],  # Bridge 18 connects Node 5 and Node 8
    19: [6, 9],  # Bridge 19 connects Node 6 and Node 9
    20: [7, 8],  # Bridge 20 connects Node 7 and Node 8
    21: [8, 9]   # Bridge 21 connects Node 8 and Node 9
}

def is_valid_jump(from_node, to_node):
    # Handle nan or None cases explicitly
    if from_node is None or to_node is None:
        return False
    if np.isnan(from_node) or np.isnan(to_node):
        return False

    # Look up valid transitions
    return to_node in valid_connections.get(from_node, [])


def is_reasonable_xy(xy_curr, xy_next, pixel_to_cm=310/22.5, criteria_cm=2):
    if xy_curr is None or xy_next is None:
        return False
    if np.any(np.isnan(xy_curr)) or np.any(np.isnan(xy_next)):
        return False

    # Calculate Euclidean distance in pixels
    distance_pixels = np.linalg.norm(xy_next - xy_curr)
    distance_cm = distance_pixels / pixel_to_cm

    return distance_cm <= criteria_cm



def smooth_tracking_data(roi_data, tracking_data, pixel_to_cm_curr_maze  = 310/22.5,criteria_for_reasonable = 2 ):
    smoothed_tracking = np.full((len(tracking_data), 2), np.nan)
    smoothed_tracking[0] = tracking_data[0]
    smoothed_roi = np.full((len(tracking_data)), np.nan) # Start with the first position
    smoothed_roi[0] = roi_data[0]
    invalid_points = 0
    for i in range(1, len(roi_data)):
        current_position = roi_data[i - 1]
        next_position = roi_data[i]
        curr_xy = tracking_data[i-1]
        next_xy = tracking_data[i]
        is_xy_reasonable = is_reasonable_xy(curr_xy, next_xy,pixel_to_cm = pixel_to_cm_curr_maze, criteria_cm = criteria_for_reasonable)
        # If the jump is invalid, we replace it with a valid position
        if is_valid_jump(current_position, next_position) == False and (current_position==next_position) == False :
            
            try:
                next_available_point_ind = np.where(np.isnan(tracking_data[i:i+1000])==False)[0][0]
            except IndexError as e: 
                break        
            next_coord = tracking_data[i+next_available_point_ind]
            
            smoothed_tracking[i] = (smoothed_tracking[i-1]+next_coord)/2 # interpolating 
            smoothed_roi[i] = (current_position)
            invalid_points += 1 
        elif is_xy_reasonable == False:
            #print(f"oh no. animal moving too fast, from{curr_xy} to {next_xy}, smoothing... ")
            smoothed_tracking[i] = (smoothed_tracking[i-1]+next_xy)/2 
            smoothed_roi[i] = (current_position)
            invalid_points += 1 
        else:
            smoothed_tracking[i] = next_xy
            smoothed_roi[i] = next_position

    percentage_invalid_transition = invalid_points/len(tracking_data)
    
    return smoothed_roi, smoothed_tracking, percentage_invalid_transition

### RECOMMENDED
def smooth_tracking_data_no_interpolation(
    roi_data,
    tracking_data,
    pixel_to_cm_curr_maze=310/22.5,
    criteria_for_reasonable=5
):
    """
    Filters out invalid transitions in ROI and tracking data based on movement criteria.
    
    Parameters:
    - roi_data: 1D array of ROI labels (e.g., integers or strings).
    - tracking_data: Nx2 array of (x, y) positions.
    - pixel_to_cm_curr_maze: conversion factor from pixels to cm.
    - criteria_for_reasonable: maximum allowed movement per frame in cm.
    
    Returns:
    - smoothed_roi: ROI data with invalid transitions set to NaN.
    - smoothed_tracking: Tracking data with unreasonable xy jumps set to NaN.
    - percentage_invalid_transition: Proportion of frames deemed invalid.
    """

    n = len(tracking_data)
    smoothed_tracking = np.full((n, 2), np.nan)
    smoothed_roi = np.full(n, np.nan)

    # Always trust the first data point
    smoothed_tracking[0] = tracking_data[0]
    smoothed_roi[0] = roi_data[0]

    invalid_points = 0
    last_valid_xy = tracking_data[0]  # Start with the first position

    for i in range(1, n):
        prev_roi = roi_data[i - 1]
        curr_roi = roi_data[i]

        prev_xy = last_valid_xy
        curr_xy = tracking_data[i]

        # Validate spatial movement
        valid_xy = is_reasonable_xy(
            prev_xy, curr_xy,
            pixel_to_cm=pixel_to_cm_curr_maze,
            criteria_cm=criteria_for_reasonable
        )

        # Validate ROI transition
        valid_roi = is_valid_jump(prev_roi, curr_roi) or (prev_roi == curr_roi)

        if valid_xy and valid_roi:
            smoothed_tracking[i] = curr_xy
            smoothed_roi[i] = curr_roi
            last_valid_xy = curr_xy
        else:
            invalid_points += 1

    percentage_invalid_transition = invalid_points / n

    return smoothed_roi, smoothed_tracking, percentage_invalid_transition


def get_poke_array(poke_in_curr, poke_out_curr, first_a_timestamp):
    #print(len(poke_in_curr), len(poke_out_curr))
    poke_temp_array = np.ndarray((np.max([len(poke_in_curr), len(poke_out_curr)]),2))
    poke_temp_array[:] = np.nan
    if len(poke_in_curr) == 0 or len(poke_out_curr) == 0:
        poke_temp_array[:] = np.nan
    elif poke_in_curr[0] < poke_out_curr[0] and abs(len(poke_in_curr)-len(poke_out_curr))==0:
        poke_temp_array[:,0] = poke_in_curr
        poke_temp_array[:,1] = poke_out_curr
    elif poke_in_curr[0] > poke_out_curr[0] and abs(len(poke_in_curr)-len(poke_out_curr))==0:
        poke_temp_array = np.vstack((poke_temp_array, [np.nan,np.nan]))
        poke_temp_array[1:,0] = poke_in_curr
        poke_temp_array[0:-1,1] = poke_out_curr
    elif poke_in_curr[0] > poke_out_curr[0] and abs(len(poke_in_curr)-len(poke_out_curr))==1:
        poke_temp_array[1:,0] = poke_in_curr
        poke_temp_array[:,1] = poke_out_curr
    elif poke_in_curr[0] < poke_out_curr[0] and abs(len(poke_in_curr)-len(poke_out_curr))==1:
        poke_temp_array[:,0] = poke_in_curr
        poke_temp_array[:-1,1] = poke_out_curr
    else:
        weird_port = True
    poke_temp_array = poke_temp_array - first_a_timestamp
    return poke_temp_array, weird_port

def get_poke_array_big(txt_dict, first_a_timestamp,num_port = 9):
    poke_big_array= np.array([])
    weird_port_list = []
    for p in range(num_port):
        poke_in_curr = txt_dict[f"poke_{p+1}"]
        poke_out_curr = txt_dict[f"poke_{p+1}_out"]
        
        #print(len(poke_in_curr), len(poke_out_curr))
        if poke_big_array.size == 0:
            if len(poke_in_curr) == 0 or len(poke_out_curr) == 0:
                continue
            poke_big_array = np.ndarray((np.max([len(poke_in_curr), len(poke_out_curr)]),2))
            poke_big_array[:] = np.nan
            if len(poke_in_curr) == 0 or len(poke_out_curr) == 0:
                poke_big_array[:] = np.nan
            elif ((poke_in_curr[0] < poke_out_curr[0]) and (abs(len(poke_in_curr)-len(poke_out_curr))==0)):
                poke_big_array[:,0] = poke_in_curr
                poke_big_array[:,1] = poke_out_curr
            elif ((poke_in_curr[0] > poke_out_curr[0]) and (abs(len(poke_in_curr)-len(poke_out_curr))==0)):
                poke_big_array = np.vstack((poke_big_array, [np.nan,np.nan]))
                poke_big_array[1:,0] = poke_in_curr
                poke_big_array[0:-1,1] = poke_out_curr
            elif ((poke_in_curr[0] > poke_out_curr[0]) and (abs(len(poke_in_curr)-len(poke_out_curr))==1)):
                poke_big_array[1:,0] = poke_in_curr
                poke_big_array[:,1] = poke_out_curr
            elif ((poke_in_curr[0] < poke_out_curr[0]) and (abs(len(poke_in_curr)-len(poke_out_curr))==1)):
                poke_big_array[:,0] = poke_in_curr
                poke_big_array[:-1,1] = poke_out_curr
            else:
                weird_port_list.append(p)
            poke_big_array = np.column_stack((poke_big_array-first_a_timestamp, np.tile(p+1, len(poke_big_array))))
        else:
            poke_temp_array = np.ndarray((np.max([len(poke_in_curr), len(poke_out_curr)]),2))
            poke_temp_array[:] = np.nan
            if len(poke_in_curr) == 0 or len(poke_out_curr) == 0:
                continue
            elif poke_in_curr[0] < poke_out_curr[0] and abs(len(poke_in_curr)-len(poke_out_curr))==0:
                poke_temp_array[:,0] = poke_in_curr
                poke_temp_array[:,1] = poke_out_curr
            elif poke_in_curr[0] > poke_out_curr[0] and abs(len(poke_in_curr)-len(poke_out_curr))==0:
                poke_temp_array = np.vstack((poke_temp_array, [np.nan,np.nan]))
                poke_temp_array[1:,0] = poke_in_curr
                poke_temp_array[0:-1,1] = poke_out_curr
            elif poke_in_curr[0] > poke_out_curr[0] and abs(len(poke_in_curr)-len(poke_out_curr))==1:
                poke_temp_array[1:,0] = poke_in_curr
                poke_temp_array[:,1] = poke_out_curr
            elif poke_in_curr[0] < poke_out_curr[0] and abs(len(poke_in_curr)-len(poke_out_curr))==1:
                poke_temp_array[:,0] = poke_in_curr
                poke_temp_array[:-1,1] = poke_out_curr
            else:
                weird_port_list.append(p)
            poke_temp_array = poke_temp_array - first_a_timestamp
            poke_temp_array = np.column_stack((poke_temp_array, np.tile(p+1, len(poke_temp_array))))
            
            poke_big_array = np.vstack((poke_big_array, poke_temp_array))

        
    poke_big_array = poke_big_array[np.argsort(poke_big_array[:,0])]



    return poke_big_array, weird_port_list

def get_led(txt_dict, event_array, poke_dict_keys, event_dict, sampling_hz = 40):
    all_poke_timestamps = []
    for key in poke_dict_keys:
        all_poke_timestamps.extend(txt_dict[key])
    all_poke_timestamps = np.array(sorted(all_poke_timestamps))
    LED_array = np.zeros(int(event_array[-1,0]))
    if txt_dict['Task_name'] == 'ABCD_normal':
        LED_array = np.zeros(int(event_array[-1,0]))
    elif  txt_dict['Task_name'] == 'ABCD_withopto':
        light_status_list = []
        light_status_list.extend(np.where(event_array == event_dict['LED_on'])[0])
        light_status_list.extend(np.where(event_array == event_dict['LED_off'])[0])
        light_status_list = sorted(light_status_list)
        for sta in range(len(light_status_list)):
            
            if event_array[light_status_list[sta]][1] == event_dict['LED_on']:
                led_start = event_array[light_status_list[sta]][0]
                if sta == (len(light_status_list)-1):
                    
                    LED_array[int(led_start):] = 1 
                    break
                
                led_end = event_array[light_status_list[sta+1]][0]
                LED_array[int(led_start):int(led_end)] = 1 
            else:
                continue

    elif  txt_dict['Task_name'] in ['ABCD_withopto_v2', 'ABCD_withopto_v3', 'ABCD_withopto_v4']:
        light_status_list = []
        light_status_list.extend(np.where(event_array == event_dict['LED_on'])[0])
        light_status_list.extend(np.where(event_array == event_dict['LED_off'])[0])
        light_status_list = sorted(light_status_list)
        for sta in range(len(light_status_list)):
            if event_array[light_status_list[sta]][1] == event_dict['LED_on']:
                led_start = event_array[light_status_list[sta]][0]
                if sta == (len(light_status_list)-1):
                    
                    if len(LED_array) - int(led_start) < 180*sampling_hz:
                        LED_array[int(led_start):] = 1 
                    else:
                        try:
                            closest_timestamp = all_poke_timestamps[np.where(all_poke_timestamps>led_start+180*sampling_hz)[0][0]]
                            LED_array[int(led_start):int(closest_timestamp)] = 1 
                        except:
                            if all_poke_timestamps[-1] - led_start < 180*sampling_hz: 
                                LED_array[int(led_start):] = 1 
                            else:
                                print("SOMEHOW GET_LED FAILED!!!") 
                    break
                

                led_end = event_array[light_status_list[sta+1]][0]
                if led_end-led_start <= 180*sampling_hz: 
                    LED_array[int(led_start):int(led_end)] = 1  
                else:
                    try:
                        closest_timestamp = all_poke_timestamps[np.where(all_poke_timestamps>led_start+180*sampling_hz)[0][0]]
                        LED_array[int(led_start):int(closest_timestamp)] = 1 
                    except:
                        if all_poke_timestamps[-1] - led_start < 180*sampling_hz: 
                            LED_array[int(led_start):] = 1 
                        else:
                            print("SOMEHOW GET_LED FAILED!!!")
                    
            else:
                continue

    elif  txt_dict['Task_name'] == 'ABCD_withopto_v5':
        last_light_off_timestamp = np.max(txt_dict['LED_off'] + txt_dict['LED_forced_off'])
        for sta in range(len(txt_dict['LED_on'])):
            led_start = txt_dict['LED_on'][sta]
            if led_start > last_light_off_timestamp:
                led_end = len(LED_array) ## the session ended with light being on 
                LED_array[int(led_start):int(led_end)] = 1 
                
            else:
                
                try:
                    led_end = txt_dict['LED_off'][np.where(np.array(txt_dict['LED_off'])>led_start )[0][0]]
                    if led_end - led_start <= 60 * sampling_hz:
                        LED_array[int(led_start):int(led_end)] = 1 
                    else: 
                        light_forced_off =txt_dict['LED_forced_off'][np.where(np.array(txt_dict['LED_forced_off']) > led_start)[0][0]]
                        LED_array[int(led_start):int(light_forced_off)] = 1 

                except:
                    ### cases where the last time the time goes on, it was forced off 
                    
                    light_forced_off =txt_dict['LED_forced_off'][np.where(np.array(txt_dict['LED_forced_off']) > led_start)[0][0]]
                    LED_array[int(led_start):int(light_forced_off)] = 1 
                

            

    else:
        print(f"{txt_dict['Task_name']} not written yet")
    
    return LED_array

    #np.save(output_filepath[i], LED_array)

def get_led_state_info(led_array, trial_time_array):
    trial_time_array_add = trial_time_array
    trial_time_array_flat = trial_time_array_add.flatten()
    output_list = []
    for re in range(len(trial_time_array_flat)-1):
        if np.isnan(trial_time_array_flat[re+1]):
            break
        if np.mean(led_array[int(trial_time_array_flat[re]):int(trial_time_array_flat[re+1])])> 0.5:
            output_list.append(1)
        else:
            output_list.append(0)
    if len(output_list)!=len(trial_time_array_flat):
        error = len(trial_time_array_flat) - len(output_list)
        for e in range(error):
            output_list.append(np.nan)
    output_array = np.array(output_list).reshape((len(trial_time_array_add), len(trial_time_array_add[0])))
    output_array[:,-1] = 0
    return output_array
