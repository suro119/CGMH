import numpy as np
import pickle as pkl

glove_path='../../data/glove/glove.840B.300d.txt'
dict_path='../../data/1-billion/dict.pkl'
emb_path='../../data/1-billion/emb.pkl'

f = open(glove_path)
emb = []

# Convert Glove Dataset (word -> emb) into a dictionary
for line in f:
    line = line.strip().split()
    word = line[0]
    word_emb = np.array([float(x) for x in line[1:]])
    emb.append([word, word_emb])

emb = dict(emb)

# 
word_dict, id_dict = pkl.load(open(dict_path))
word_list = word_dict.keys()
emb_small = []
emb_small_id = []

for i in range(len(word_list)):
    word = id_dict[i]