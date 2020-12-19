from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import time

import numpy as np
import tensorflow as tf
from tensorflow.python.layers import core as layers_core
import reader
from config import config
config=config()
from tensorflow.python.client import device_lib
import os
os.environ['CUDA_VISIBLE_DEVICES']=config.GPU

from utils import *

def data_type():
  return tf.float32

class LangModel():
    def __init__(self, ckpt_path):
        self.ckpt_path = ckpt_path
        self.model = tf.keras.Sequential([
            # B: batch size
            # M: max sentence length
            # H: hidden size
            # V: vocab size
            tf.keras.layers.Masking(mask_value=config.dict_size+3),  # B, M
            tf.keras.layers.Embedding(config.vocab_size, config.hidden_size),  # B, M, H
            tf.keras.layers.LSTM(config.hidden_size, unit_forget_bias=False, return_sequences=True),  # B, M, H
            tf.keras.layers.LSTM(config.hidden_size, unit_forget_bias=False, return_sequences=True),  # B, M, H
            tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(config.vocab_size))  # B, M, V
        ])
        self.softmax = tf.keras.layers.Softmax()
    
    def compile(self):
        def loss(labels, logits):
            return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)
            # mask=tf.sequence_mask(lengths=sequence_length, maxlen=config.num_steps, dtype=data_type())
            # return tfa.seq2seq.sequence_loss(logits, labels, mask, average_across_timesteps=True, average_across_batch=True)
        self.model.compile(optimizer='adam', loss=loss)

    def run(self, train_dataset, val_dataset, epochs, batch_size):
        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=self.ckpt_path,
            monitor='val_loss',
            mode='min',
            save_best_only=True)

        early_callback = tf.keras.callbacks.EarlyStopping()

        self.model.fit(
            train_dataset, 
            epochs=epochs, 
            batch_size=batch_size,
            validation_data=val_dataset, 
            callbacks=[checkpoint_callback, early_callback]
        )

    def predict(self, input):
        if isinstance(input, list):
            input = np.array(input)
        logits = self.model.predict(input)
        probs = self.softmax(logits).numpy()

        return probs

    def restore(self):
        if not os.path.exists(self.ckpt_path):
            raise FileNotFoundError('File {} does not exist'.format(self.ckpt_path))
        self.model = tf.keras.models.load_model(self.ckpt_path, compile=False)


def main():
    if config.mode == 'forward' or config.mode == 'use':
        m_forward = LangModel(config.forward_save_path)
    if config.mode == 'backward' or config.mode == 'use':
        m_backward = LangModel(config.backward_save_path)
    
    if config.mode == 'forward':
        train_data, train_sequence_length, test_data, test_sequence_length = reader.read_data(config.data_path, config.num_steps)
        m_forward.compile()
        m_forward.run(train_data, test_data, config.max_epoch, config.batch_size)

    if config.mode == 'backward':
        train_data, train_sequence_length, test_data, test_sequence_length = reader.read_data(config.data_path, config.num_steps, is_backward=True)
        m_backward.compile()
        m_backward.run(train_data, test_data, config.max_epoch, config.batch_size)
            
    # CGMH initialization: line 202-254 (수로)
    if config.mode=='use':
        #CGMH sampling for key_gen
        m_forward.restore()
        m_backward.restore()
        # Initially set to False
        if config.keyboard_input == True:
            key_input = raw_input('Please enter a sentence\n')
            if key_input == '':
                dataset, sequence_lengths, sta_vec_list = reader.read_data_use(config.use_data_path, config.num_steps)
            else:
                key_input = key_input.split()
                key_input = sen2id(key_input)
                sta_vec = list(np.zeros([config.num_steps-1]))
                for i in range(len(key_input)):
                    sta_vec[i] = 1
                dataset, sequence_lengths = reader.array_data([key_input], config.num_steps, config.dict_size)
        else:
            dataset, sequence_lengths, sta_vec_list = reader.read_data_use(config.use_data_path, config.num_steps)
        config.batch_size = 1

    for sen_id, data in enumerate(dataset.as_numpy_iterator()): # For each sentence in the list of sentences
        input = data[0]
        sequence_length = sequence_lengths[sen_id]

        if config.keyboard_input == False:
            sta_vec = sta_vec_list[sen_id%(config.num_steps-1)]
        print(sta_vec)

        pos = 0
        outputs = []
        output_p = []
        for iter in range(config.sample_time):
            print('\n\n-------------------Iter: {}--------------------'.format(iter))
            config.sample_prior = [1, 10.0/sequence_length, 1, 1]
            if iter % 20 < 10:
                config.threshold = 0
            else:
                config.threshold = 0.5
            ind = pos % (sequence_length)
            action = choose_action(config.action_prob)
            # Print input sentence
            print(input)
            print(' '.join(id2sen(input)))
            # If keyword chosen, 
            if sta_vec[ind] == 1 and action in [0, 2]:
                action = 3

            # Word replacement (action: 0)
            sequence_length_minus = sequence_length - 1
            if action == 0 and ind < sequence_length_minus:
                prob_old = m_forward.predict([input])

                temp = 1
                for j in range(sequence_length_minus):
                    temp *= prob_old[0][j][input[j+1]] # conditional probability of when given a sentence until [0, j] --> j+1 th probability
                temp *= prob_old[0][j+1][config.dict_size + 1] # 50000 end of sentence's probability
                prob_old_prob = temp
                
                # Generate input_forward and input_backward
                input_forward, input_backward, sequence_length_forward, sequence_length_backward = cut_from_point([input], [sequence_length], ind, mode=action) # mode = 0, generate backwards
                
                prob_forward = m_forward.predict(input_forward)[0, ind%sequence_length_minus, :] 
                prob_backward = m_backward.predict(input_backward)[0, sequence_length_minus-ind%sequence_length_minus, :]
                prob_mul = prob_forward * prob_backward

                input_candidate, sequence_length_candidate = generate_candidate_input([input], [sequence_length], ind, prob_mul, config.search_size, mode=action)
                prob_candidate_pre = m_forward.predict(input_candidate)
                
                # Final candidate
                prob_candidate = []
                for i in range(config.search_size): # 100
                    temp = 1
                    for j in range(sequence_length_minus):
                            temp *= prob_candidate_pre[i][j][input_candidate[i][j+1]]
                    temp *= prob_candidate_pre[i][j+1][config.dict_size + 1]
                    prob_candidate.append(temp)
                
                # Make it into numpy array
                prob_candidate = np.array(prob_candidate)

                prob_candidate_norm = normalize(prob_candidate)
                prob_candidate_ind = sample_from_candidate(prob_candidate_norm)
                prob_candidate_prob = prob_candidate[prob_candidate_ind]

                if input_candidate[prob_candidate_ind][ind + 1] < config.dict_size and (prob_candidate_prob > prob_old_prob * config.threshold or just_acc() == 0):
                    input = input_candidate[prob_candidate_ind] # cut at index = prob_candidate_ind
                # alpha of replacement = 1
                
                pos += 1
                print ('action: 0', 1, prob_old_prob, prob_candidate_prob, prob_candidate_norm[prob_candidate_ind])
                if ' '.join(id2sen(input)) not in output_p:
                    outputs.append([' '.join(id2sen(input)), prob_old_prob])


            # Word insertion (action:1)
            if action == 1:
                # Sentence cannot be longer than config.num_steps words
                if sequence_length >= config.num_steps:
                    action = 3
                else:
                    input_forward, input_backward, sequence_length_forward, sequence_length_backward = cut_from_point([input], [sequence_length], ind, mode=action)
                    # Returns the list of probabilities for each w, where each element is a prbability of (sentence upto 'ind' + w)
                    prob_forward = m_forward.predict(input_forward)[0, ind % (sequence_length-1), :]
                    # Returns the list of probabilities for each w, where each element is a prbability of (w + sentence starting from ind+1)
                    prob_backward = m_backward.predict(input_backward)[0, sequence_length-1-(ind % (sequence_length-1)),:]
                    prob_mul = prob_forward * prob_backward
                    input_candidate, sequence_length_candidate = generate_candidate_input([input], [sequence_length], ind, prob_mul, config.search_size, mode=action)
                    prob_candidate_pre = m_forward.predict(input_candidate)

                    prob_candidate = []
                    # For each candidate sentence, calculate the probability of the candidate
                    for i in range(config.search_size):
                        tem = 1
                        # Continuously multiply the probability of the j+1th word given the sentence upto 
                        # the jth word
                        for j in range(sequence_length_candidate[0]-1):
                                tem *= prob_candidate_pre[i][j][input_candidate[i][j+1]]
                        # Multiply the probability of EOS token at the end
                        tem *= prob_candidate_pre[i][j+1][config.dict_size+1]
                        prob_candidate.append(tem)
                    prob_candidate = np.array(prob_candidate) * config.sample_prior[1]  # config.sample_prior = 10.0/(sentence length)
                    prob_candidate_norm = normalize(prob_candidate)

                    prob_candidate_ind = sample_from_candidate(prob_candidate_norm)
                    prob_candidate_prob = prob_candidate[prob_candidate_ind] # Unnormalized candidate probability

                    prob_old = m_forward.predict([input])

                    # Calculate the probability of the original sentence and save in 'prob_old_prob'
                    tem = 1
                    for j in range(sequence_length-1):
                        tem *= prob_old[0][j][input[j+1]]
                    tem *= prob_old[0][j+1][config.dict_size+1]

                    prob_old_prob = tem

                    alpha = min(1, prob_candidate_prob * config.action_prob[2] / (prob_old_prob * config.action_prob[1] * prob_candidate_norm[prob_candidate_ind]))
                    print ('action:1',alpha, prob_old_prob, prob_candidate_prob, prob_candidate_norm[prob_candidate_ind])

                    # Save old sentence in output with its probability
                    if ' '.join(id2sen(input)) not in output_p:
                        outputs.append([' '.join(id2sen(input)), prob_old_prob])

                    # 1. Must be chosen by alpha
                    # 2. Must have a valid value in insertion position
                    # 3. Must have a probability higher than 0.5 * (original sentence probability)
                    if (choose_action([alpha, 1-alpha]) == 0 and
                        input_candidate[prob_candidate_ind][ind+1] < config.dict_size and
                        prob_candidate_prob > prob_old_prob * config.threshold):
                        input = input_candidate[prob_candidate_ind]  # Insert
                        sequence_length += 1
                        pos += 2  # Skip inserted word
                        sta_vec.insert(ind, 0.0)
                        del(sta_vec[-1])  # Delete sta_vec[config.num_steps], which should be invalid
                    # No insertion if conditions are not met
                    else:
                        action = 3


            # Word deletion (action: 2)
            if action == 2 and ind < sequence_length - 1:
                if sequence_length <= 2:
                    # skip word
                    action = 3
                else:
                    prob_old = m_forward.predict([input])
                        
                tem = 1
                for j in range(sequence_length - 1):
                    tem *= prob_old[0][j][input[j+1]]
                tem *= prob_old[0][j+1][config.dict_size+1]
                prob_old_prob = tem

                # find the probability of deleted sentence 
                input_candidate, sequence_length_candidate = generate_candidate_input([input], [sequence_length], ind, None, config.search_size, mode=2)
                prob_new = m_forward.predict(input_candidate)

                # reset tem
                tem = 1
                for j in range(sequence_length_candidate[0]-1):
                    tem *= prob_new[0][j][input_candidate[0][j+1]]
                tem *= prob_new[0][j+1][config.dict_size+1]
                prob_new_prob = tem

                input_forward, input_backward, sequence_length_forward, sequence_length_backward = cut_from_point([input], [sequence_length], ind, mode=0)
                prob_forward = m_forward.predict(input_forward)[0, ind % (sequence_length-1),:]
                prob_backward = m_backward.predict(input_backward)[0, sequence_length - 1 - ind%(sequence_length-1),:]
                prob_mul = (prob_forward * prob_backward)

                # find the probability of candidate words that may be replaced in the deleted space
                input_candidate, sequence_length_candidate = generate_candidate_input([input], [sequence_length], ind, prob_mul, config.search_size, mode=0)
                prob_candidate_pre = m_forward.predict(input_candidate)

                prob_candidate = []
                for i in range(config.search_size):
                    # reset tem
                    tem = 1
                    for j in range(sequence_length-1):
                        tem *= prob_candidate_pre[i][j][input_candidate[i][j+1]]
                    tem *= prob_candidate_pre[i][j+1][config.dict_size+1]
                    prob_candidate.append(tem)
                prob_candidate = np.array(prob_candidate)

                # alpha is acceptance ratio of current proposal
                prob_candidate_norm = normalize(prob_candidate)

                # among the 100 candidates, if the word to be deleted is not included in the candidates, set alpha to 0    
                if input in input_candidate:
                    for candidate_ind in range(len(input_candidate)):
                        if input in input_candidate[candidate_ind: candidate_ind+1]:
                            break
                        pass
                    # calculate the acceptance rate
                    numer = prob_candidate_norm[candidate_ind] * prob_new_prob * config.action_prob[1]
                    denom = config.action_prob[2] * prob_old_prob
                    alpha = min(numer/denom, 1)
                else:
                    pass
                    alpha = 0

                if ' '.join(id2sen(input)) not in output_p:
                    outputs.append([' '.join(id2sen(input)), prob_old_prob])

                # if deletion of word fits under acceptance rate, delete
                if choose_action([alpha, 1 - alpha]) == 0 and (prob_new_prob > prob_old_prob * config.threshold or just_acc() == 0):
                    input = np.concatenate([input[:ind+1], input[ind+2:], input[:1] * 0 + config.dict_size + 3])
                    # deleted
                    sequence_length -= 1
                    pos += 0
                    del(sta_vec[ind])
                    sta_vec.append(0)
                # if not, skip
                else:
                    action = 3

            # Skip (action: 3)
            if action == 3:
                pos += 1

            if outputs != []:
                output_p.append(outputs[-1][0])

        for num in range(config.max_length, 0, -1):
            outputss = [x for x in outputs if len(x[0].split()) == num]
            if len(outputss) == 0:
                continue
            print(num, outputss)
            # if outputs != []:
            #     continue
            # if outputss == []:
            #     outputss.append([' '.join(id2sen(input)), 1])
            outputss = sorted(outputss, key=lambda x: x[1])[::-1]  # reverse list using [::-1]
            outputss = outputss[:5]  # get 5 most likely sentneces of each length
            outputss = map(lambda t: t[0], outputss)
            with open(config.use_output_path, 'a') as g:
                for sent in outputss:
                    g.write(sent + '\n')

            
        

if __name__ == "__main__":
    main()
