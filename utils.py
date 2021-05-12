import math
import numpy as np
import scipy.signal as signal


def elbow_angle(l_array, r_array, keypoint_coords):

    '''
    Keypoints Indexes
    Id	Part
    5	leftShoulder
    6	rightShoulder
    7	leftElbow
    8	rightElbow
    9	leftWrist
    10	rightWrist
    '''
    # print(f'pose scrores: {pose_scores}\n keypoint scroese: {keypoint_scores[:,5:11]}\n keypoints coord: {keypoint_coords[:,5:11]}')

    l_wrist = keypoint_coords[9]
    l_elbow = keypoint_coords[7]
    l_shoulder = keypoint_coords[5]
    r_wrist = keypoint_coords[10]
    r_elbow = keypoint_coords[8]
    r_shoulder = keypoint_coords[6]

    # limb vectors
    l_elbow_wrist = l_wrist - l_elbow
    l_elbow_should = l_shoulder - l_elbow
    r_elbow_wrist = r_wrist - r_elbow
    r_elbow_should = r_shoulder - r_elbow

    # dot product between two limb vectors
    l_elbow_dot = np.dot(l_elbow_wrist, np.transpose(l_elbow_should))
    r_elbow_dot = np.dot(r_elbow_wrist, np.transpose(r_elbow_should))

    # norms of limb vectors
    l_ew_norm = np.linalg.norm(l_elbow_wrist)
    l_es_norm = np.linalg.norm(l_elbow_should)
    r_ew_norm = np.linalg.norm(r_elbow_wrist)
    r_es_norm = np.linalg.norm(r_elbow_should)

    # angle of elbow
    l_elbow_angle = math.degrees(math.acos(l_elbow_dot / (l_ew_norm * l_es_norm)))
    r_elbow_angle = math.degrees(math.acos(r_elbow_dot / (r_ew_norm * r_es_norm)))

    # append existing historic of angles
    l_array.append(l_elbow_angle)
    r_array.append(r_elbow_angle)

    return l_array, r_array


def elbow_engagement(l_array, r_array):
    l_diff, r_diff = 0, 0

    l_min_angle, r_min_angle = 999, 999
    l_max_angle, r_max_angle = -999, -999

    # finding which was the biggest and smallest elbow angles over the past frames_window frames
    for i in range(len(l_array)):
        if l_max_angle < l_array[i]:
            l_max_angle = l_array[i]
        elif l_min_angle > l_array[i]:
            l_min_angle = l_array[i]

        if r_max_angle < r_array[i]:
            r_max_angle = r_array[i]
        elif r_min_angle > r_array[i]:
            r_min_angle = r_array[i]

    # comparing the angle amplitude of the elbows over the past frames_window frames
    l_diff = np.abs(l_max_angle - l_min_angle)
    r_diff = np.abs(r_max_angle - r_min_angle)

    return l_diff, r_diff



def _calculate_movement(sequence):
    '''
    Calulates movement of a single body part given its trajectory (i.e. positions over the time frame)

    '''
    total_movement = 0
    for i in range(len(sequence)-1):
        total_movement += np.linalg.norm(sequence[i+1] - sequence[i], ord=2)
    return total_movement


def upper_body_movement(keypoint_coords_history):
    '''
    Average upper body movement = Mean of individual movements of 6 parts (2 shoulders, 2 elbows, 2 wrists)

    '''

    keypoint_coords_history = np.array(keypoint_coords_history)

    '''
    Keypoints Indexes
    Id	Part
    5	leftShoulder
    6	rightShoulder
    7	leftElbow
    8	rightElbow
    9	leftWrist
    10	rightWrist
    '''

    l_shoulder_sequence = keypoint_coords_history[:,5,:]
    l_shoulder_movement = _calculate_movement(l_shoulder_sequence)

    r_shoulder_sequence = keypoint_coords_history[:,6,:]
    r_shoulder_movement = _calculate_movement(r_shoulder_sequence)

    l_elbow_sequence = keypoint_coords_history[:,7,:]
    l_elbow_movement = _calculate_movement(l_elbow_sequence)

    r_elbow_sequence = keypoint_coords_history[:,8,:]
    r_elbow_movement = _calculate_movement(r_elbow_sequence)

    l_wrist_sequence = keypoint_coords_history[:,9,:]
    l_wrist_movement = _calculate_movement(l_wrist_sequence)

    r_wrist_sequence = keypoint_coords_history[:,10,:]
    r_wrist_movement = _calculate_movement(r_wrist_sequence)

    avg_upper_body_movement = np.mean([l_shoulder_movement, r_shoulder_movement,
                                       l_elbow_movement, r_elbow_movement,
                                       l_wrist_movement, r_wrist_movement])

    return avg_upper_body_movement


def scale(avg_upper_body_movement, keypoint_coords):
    '''
    Performs scaling body movement so that it's same whether you are up close or far from the camera.
    Considers the shoulder-to-shoulder distance as the scaling factor.

    '''
    shoulder_line_length = np.linalg.norm(keypoint_coords[5]-keypoint_coords[6], ord=2)
    return avg_upper_body_movement / shoulder_line_length



###############################################################################
# Filter function for the mood score

def apply_filter(input_signal_patch, history_window_size, filter_type='gaussian', std_dev=10):

    if len(input_signal_patch) < history_window_size:
        return input_signal_patch[-1]

    if filter_type == 'average':
        filtered_sample = np.mean(input_signal_patch)

    if filter_type == 'gaussian':
        ga = signal.gaussian(history_window_size*2-1,std_dev)

        ga_half = ga[:history_window_size]
        ga_half = ga_half/ga_half.sum()

        filtered_sample = np.dot(input_signal_patch, ga_half)

    return filtered_sample

###############################################################################
# Heuristics

def mood_heuristic(emotion_probabilities):
    '''
    mood score = emotion weight * highest emotion probability

    where,
        emotion weight is in range [0,1]

    '''

    weights = {0: 0.1, # Angry
               1: 0.1, # Disgust
               2: 0.1, # Fear
               3: 1.0, # Happy
               4: 0.1, # Sad
               5: 0.9, # Surprise
               6: 0.8  # Neutral
               }

    emotion_idx = np.argmax(emotion_probabilities)

    mood_score = emotion_probabilities[emotion_idx] * weights[emotion_idx]
    return mood_score.squeeze()


def engagement_heuristic(elbow_angle_diff, upper_body_movement):
    '''
    engagement score = f1(elbow angle diff) * f2(upper body movement)

    where,
        elbow angle diff = 11 to 25 when the player is static
                         = 50 to 180 when the player is playing

        upper body movement = around 1.0 when the player is static
                            = upto 4.0 when the player is playing
    '''

    engagement_score = _f1(elbow_angle_diff) * _f2(upper_body_movement)
    return engagement_score


def _f1(elbow_angle_diff):
    '''
    Non-linear squashing function for elbow angle diff
        > Modified sigmoid
    '''
    center_point = 30  # Found these values through a small experiment
    factor = 1/2       #
    return 1/(1+np.exp(-(elbow_angle_diff-center_point)*factor))


def _f2(upper_body_movement):
    '''
    Non-linear squashing function for upper body movement
        > Modified sigmoid
    '''
    center_point = 2.5  # Found these values through a small experiment
    factor = 3          #
    return 1/(1+np.exp(-(upper_body_movement-center_point)*factor))