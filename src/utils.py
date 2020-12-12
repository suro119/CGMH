import numpy as np
from config import config
from copy import copy

config = config()


import sys
sys.path.insert(0, config.skipthoughts_path) # '../skip_thoughts'
sys.path.insert(0, config.dict_path)
sys.path.insert(0, '../utils/dict_emb') # Adjust according to the new directory structure

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
    input_new = np.zeros([batch_size, num_steps]) + config.dict_size + 3 # config.dict_size + 3: Mask token
    target_new = np.zeros([batch_size, num_steps]) + config.dict_size + 3

    for i in range(batch_size):
        length = sequence_length[i] - 1

        # Set target_new
        for j in range(length):
            target_new[i][j] = target[i][length-1-j]
        target_new[i][length+1] = config.dict_size + 2 # EOS
        
        # Set input_new
        input_new[i][0] = config.dict_size + 2  # BOS
        for j in range(length):
            input_new[i][j+1] = input[i][length-j]
        input_new[i][length+1] = config.dict_size + 2  # EOS
        
    return input_new.astype(np.int32), sequence_length.astype(np.int32), target_new.astype(np.int32)


# Cut at 'ind'
# Generate backwards for replacement and insertion only.
def cut_from_point(input, sequence_length, ind, mode = 0): # mode could be 0 or action no.
    if isinstance(input, list):
        input = np.array(input)

    if isinstance(sequence_length, list):
        sequence_length = np.array(sequence_length)

    batch_size = input.shape[0]
    num_steps = input.shape[1]

    # Initialize to Mask
    input_forward = np.zeros([batch_size, num_steps]) + config.dict_size + 3 # dict_size = 50000
    input_backward = np.zeros([batch_size, num_steps]) + config.dict_size + 3

    sequence_length_forward = np.zeros([batch_size])
    sequence_length_backward = np.zeros([batch_size])

    for i in range(batch_size):
        # Set to BOS
        input_forward[i][0] = config.dict_size + 2
        input_backward[i][0] = config.dict_size + 2
        length = sequence_length[i] - 1

        # Set input_forward & sequence_length_forward
        for j in range(ind):
            input_forward[i][j+1] = input[i][j+1]
        input_forward[i][ind+1] = config.dict_size + 1  # EOS
        sequence_length_forward[i] = ind + 1

        # Set input_backward & sequence_length_backward
        if mode == 0:
            for j in range(length-ind-1):
                input_backward[i][j+1] = input[i][length-j]
            input_backward[i][length-ind] = config.dict_size + 1  # EOS
            sequence_length_backward[i] = length - ind
        elif mode == 1:
            for j in range(length-ind):
                input_backward[i][j+1] = input[i][length-j]
            input_backward[i][length-ind+1] = config.dict_size + 1  # EOS
            sequence_length_backward[i] = length - ind + 1
    
    return input_forward.astype(np.int32), input_backward.astype(np.int32), sequence_length_forward.astype(np.int32), sequence_length_backward.astype(np.int32)



def generate_candidate_input(input, sequence_length, ind, prob, search_size, mode=0):
    input_new = np.array([input[0]] * search_size)
    sequence_length_new = np.array([sequence_length[0]] * search_size)
    length = sequence_length[0] - 1

    # If not deleting, 'ind_token' is the list of 'search_size' word indices with the highest probability
    if mode != 2:
        ind_token = np.argsort(prob[:config.dict_size])[-search_size:]
    
    # If deleting
    if mode == 2:
        # Replace input[:, 'ind'+1:length-1] with input[:, 'ind'+2:length]
        # In other words, remove input[:, 'ind'+1]
        # 'ind' + 1 to account for BOS token
        for i in range(sequence_length[0]-ind-2):
            input_new[:, ind+i+1] = input_new[:, ind+i+2]
        # Add Mask token to leftover elements
        for i in range(sequence_length[0]-1, config.num_steps-1):
            input_new[:, i] = config.dict_size + 3
        sequence_length_new = sequence_length_new - 1
        return input_new[:1], sequence_length_new[:1]

    # If inserting, make space for the new word to be inserted
    if mode == 1:
        for i in range(0, sequence_length_new[0]-1-ind):
            input_new[:, sequence_length_new[0]-i] = input_new[:,  sequence_length_new[0] - 1 - i]
        sequence_length_new = sequence_length_new + 1

    # Insert the candidate words
    for i in range(search_size):
        input_new[i][ind+1] = ind_token[i]

    return input_new.astype(np.int32), sequence_length_new.astype(np.int32)



def sample_from_candidate(prob_candidate):
    return choose_action(normalize(prob_candidate))

# Used to choose/sample an action given a list of probabilities
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