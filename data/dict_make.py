from collections import Counter
import pickle as pkl

dict_size = 30000

with open('1-billion.txt', 'r') as f:
    sentence = f.read()
counter = Counter(list(map(lambda w: w.lower(), sentence.split())))
counter = counter.most_common(dict_size)

words = list(zip(*counter))[0]

word2id = dict(zip(words, range(dict_size)))
id2word = dict(zip(range(dict_size), words))

with open('dict.pkl', 'wb') as f:
    pkl.dump([word2id, id2word], f)

# with open('dict.pkl', 'rb') as f:
#     data = pkl.load(f)
#     print(data[0])
#     print(data[1])
