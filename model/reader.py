from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import pickle as pkl
from config import config
config = config()
from utils import *
import sys
sys.path.insert(0, config.dict_path)
# from dict_use import * # Dictionary is not used in key_gen
tt_proportion = 0.9

# Returns the callable Dataset Class given the list of sentences
def array_data(data, max_length, dict_size, is_backward=False, shuffle=False):
    max_length_m1 = max_length - 1
    if shuffle == True:
        np.random.shuffle(data)

    # Record length of each sentence in a list
    sequence_length_pre = np.array([len(line) for line in data]).astype(np.int32)
    sequence_length = []

    for item in sequence_length_pre:
        if item >= max_length_m1:
            sequence_length.append(max_length)
        else:
            sequence_length.append(item + 1)  # Adding 1 to account for BOS token or EOS token
    sequence_length = np.array(sequence_length)
    for i in range(len(data)):
        if len(data[i]) >= max_length_m1:
            data[i] = data[i][:max_length_m1]
            data[i].append(dict_size + 1)
        else:
            data[i].append(dict_size + 1)
            for j in range(max_length_m1 - len(data[i])):
                data[i].append(dict_size + 3)  # Add masks
        data[i].append(dict_size + 1)  # Appends EOS token at the end of the sentence
    
    target = np.array(data).astype(np.int32)
    # Concatenates BOS token along with the rest of the sentence
    input = np.concatenate([np.ones([len(data), 1]) * (dict_size + 2), target[:, :-1]], axis=1).astype(np.int32)

    if is_backward:
        input, sequence_length, target = reverse_seq(input, sequence_length, target)

    # input: BOS + Sentence
    # target: Sentence +  EOS
    dataset = tf.data.Dataset.from_tensor_slices((input, target))
    return dataset, sequence_length

# Data reader for RNN Training
def read_data(file_name, max_length, dict_size=config.dict_size, is_backward=False):
	if file_name[-3:]=='pkl':
	    data = pkl.load(open(file_name))
	else:
	    with open(file_name) as f:
	        data = []
	        for line in f:
	            data.append(sen2id(line.strip().split()))
    # Divide dataset into train and test data using 'tt_proportion'
	train_data, train_sequence_length = array_data(data[ : int(len(data) * tt_proportion)], max_length, dict_size, is_backward=is_backward, shuffle=True)
	test_data, test_sequence_length = array_data(data[int(len(data) * tt_proportion) : ], max_length, dict_size, is_backward=is_backward, shuffle=True)
	return train_data, train_sequence_length, test_data, test_sequence_length

# Data reader for CGMH iterations
def read_data_use(file_name, max_length, dict_size=config.dict_size, is_backward=False):
    if file_name[-3:] == 'pkl':
        data = pkl.load(open(file_name))
        sta_vec_list = [list(np.zeros([config.num_steps-1]))] * len(data)                     
    else:
        with open(file_name) as f:
            data = []
            sta_vec_list = []
            j = 0
            for line in f:
                sta_vec = list(np.zeros([config.num_steps-1]))
                line = sen2id(line.strip().lower().split())
                key = choose_key(line, config.key_num)
                for i in range(len(key)):
                    sta_vec[i] = 1
                sta_vec_list.append(sta_vec)
                data.append(key)
    dataset, sequence_length = array_data(data, max_length, dict_size, is_backward=is_backward)
    return dataset, sequence_length, sta_vec_list
    
# local function only used in reader.py to choose keys
def choose_key(line, num):
    ind_list = range(len(line))
    np.random.shuffle(ind_list)
    ind_list = ind_list[:num]
    ind_list.sort()
    tem = []
    for ind in ind_list:
        tem.append(line[ind])
    return tem
