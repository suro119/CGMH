import numpy as np
from config import config
from copy import copy

config = config()


import sys
sys.path.insert(0, config.skipthoughts_path) # '../skip_thoughts'
sys.path.insert(0, config.dict_path)
sys.path.insert(0, './dict_use') # Adjust according to the new directory structure

from dict_use import dict_use
dict_use = dict_use(config.dict_path)
sen2id = dict_use.sen2id
id2sen = dict_use.id2sen


def normalize(x, e = 0.05):
    temp = copy(x)
    if max(temp) == 0:
        temp += e
    return temp / temp.sum()


# Used to generate backward inputs or input candidates.
# Most of the time only uses the first returning value.
def reverse_seq(input, sequence_length, target):
    batch_size = input.shape[0]
    num_steps = input.shape[1]

    # Initialize
    input_new = np.zeros([batch_size, num_steps]) + config.dict_size + 1 # dict_size = 50000
    target_new = np.zeros([batch_size, num_steps]) + config.dict_size + 1

    for i in range(batch_size):
        length = sequence_length[i] - 1

        # Set target_new
        for j in range(length):
            target_new[i][j] = target[i][length-1-j]
        
        # Set input_new
        input_new[i][0] = config.dict_size + 2
        for j in range(length):
            input_new[i][j+1] = input[i][length-j]
        
    return input_new.astype(np.int32), sequence_length.astype(np.int32), target_new.astype(np.int32)


# Cut at 'ind'
# Generate backwards for replacement and insertion only.
def cut_from_point(input, sequence_length, ind, mode = 0): # mode could be 0 or action no.
    batch_size = input.shape[0]
    num_steps = input.shape[1]

    # Initialize
    input_forward = np.zeros([batch_size, num_steps]) + config.dict_size + 1 # dict_size = 50000
    input_backward = np.zeros([batch_size, num_steps]) + config.dict_size + 1

    sequence_length_forward = np.zeros([batch_size])
    sequence_length_backward = np.zeros([batch_size])

    for i in range(batch_size):
        input_forward[i][0] = config.dict_size + 2
        input_backward[i][0] = config.dict_size + 2
        length = sequence_length[i] - 1

        # Set input_forward & sequence_length_forward
        for j in range(ind):
            input_forward[i][j+1] = input[i][j+1]
        sequence_length_forward[i] = ind + 1

        # Set input_backward & sequence_length_backward
        if mode == 0:
            for j in range(length-ind-1):
                input_backward[i][j+1] = input[i][length-j]
            sequence_length_backward[i] = length - ind
        elif mode == 1:
            for j in range(length-ind):
                input_backward[i][j+1] = input[i][length-j]
            sequence_length_backward[i] = length - ind + 1
    
    return input_forward.astype(np.int32), input_backward.astype(np.int32), sequence_length_forward.astype(np.int32), sequence_length_backward.astype(np.int32)



def generate_candidate_input(input, sequence_length, ind, prob, search_size, mode = 0):
    input_new = np.array([input[0]] * search_size)
    sequence_length_new = np.array([sequence_length[0]] * search_size)
    length = sequence_length[0] - 1

    if mode != 2:
        ind_token = np.argsort(prob[:config.dict_size])[-search_size:]
    
    if mode == 2:
        for i in range(sequence_length[0]-ind-2):
            input_new[:, ind+i+1] = input_new[:, ind+i+2]
        for i in range(sequence_length[0]-1, config.num_steps-1):
            input_new[:, i] = input_new[:, i] * 0 + config.dict_size + 1
        sequence_length_new = sequence_length_new - 1
        return input_new[:1], sequence_length_new[:1]
    if mode == 1:
        for i in range(0, sequence_length_new[0]-1-ind):
            input_new[:, sequence_length_new[0]-i] = input_new[:,  sequence_length_new[0] - 1 - i]
        sequence_length_new = sequence_length_new + 1

    for i in range(search_size):
        input_new[i][ind+1] = ind_token[i]

    return input_new.astype(np.int32), sequence_length_new.astype(np.int32)



def sample_from_candidate(prob_candidate):
    return choose_action(normalize(prob_candidate))

# Used only once to select action from the candidates (normalized prob_candidate)
def choose_action(c):
    r = np.random.random()
    c = np.array(c)

    for i in range(1, len(c)):
        c[i] = c[i] + c[i-1]

    for i in range(len(c)):
        if c[i] >= r:
            return i



def just_acc():
    r = np.random.random()
    if r < config.just_acc_rate: # config.just_acc_rate = 0.0
        return 0
    else:
        return 1

def write_log(str, path):
    with open(path, 'a') as g:
        g.write(str + '\n')