# Import


# RNN


# CGMH initialization: line 202-254 (수로)
if config.mode=='use':
      #CGMH sampling for key_gen
      # Load RNN Model (needs upgrading to TF 2.0)
      saver_forward.restore(session, config.forward_save_path)
      saver_backward.restore(session, config.backward_save_path)

      # Initially set to False
      if config.keyborad_input == True:
            key_input = raw_input('Please enter a sentence\n')
            if key_input = '':
                  use_data = reader.read_data_use(config.use_data_path, config.num_steps)
            else:
                  key_input = key_input.split()
                  key_input = sen2id(key_input)
                  sta_vec = list(np.zeros([config.num_steps-1]))
                  for i in range(len(key_input)):
                        sta_vec[i] = 1
                  use_data = reader.array_data([key_input], config.num_steps, config.dict_size)
      else:
            use_data, sta_vec_list = reader.read_data_use(config.use_data_path, config.num_steps)
      config.batch_size = 1

for sen_id in range(len(use_data)):  # For each word in the list of keywords
      if config.keyborad_input == False:
            sta_vec = sta_vec_list[sen_id%(config.num_steps-1)]
      
      print(sta_vec)

      input, sequence_length, _ = use_data(1, sen_id) # Batch size = 1
      input_original=input[0]

      pos=0
      outputs = []
      output_p = []
      for iter in range(config.sample_time):
            pass


# CGMH replacement: line 256-303 (정언)
#word replacement (action: 0)
                        sequence_length_minus = sequence_length[0] - 1
                        if action == 0 and ind < sequence_length_minus:
                              prob_old = run_epoch(session, mtest_forward, input, sequence_length, mode='use') # Default mode seems to be 'train'
                              if config.double_LM == True: # default config.double_LM is False
                                    input_backward, _, _ = reverse_seq(input, sequence_length, input)
                                    prob_old += run_epoch(session, mtest_backward, input_backward, sequence_length, mode='use')
                                    prob_old *= 0.5

                              temp = 1
                              for j in range(sequence_length_minus):
                                    temp *= prob_old[0][j][input[0][j+1]]
                              temp *= prob_old[0][j+1][config.dict_size + 1] # 50000
                              prob_old_prob = temp

                              if sim != None: # but config.sim = None(?)
                                    similarity_old = similarity(input[0], input_original, sta_vec)
                                    prob_old_prob *= similarity_old
                              else:
                                    similarity_old = -1
                              
                              # Generate input_forward and input_backward
                              input_forward, input_backward, sequence_length_forward, sequence_length_backward = cut_from_point(input, sequence_length, ind, mode=action) # mode = 0, generate backwards
                              
                              prob_forward = run_epoch(session, mtest_forward, input_forward, sequence_length_forward, mode='use')[0, ind%sequence_length_minus, :]
                              prob_backward = run_epoch(session, mtest_backward, input_backward, sequence_length_backward, mode='use')[0, sequence_length_minus-ind%sequence_length_minus, :]
                              prob_mul = prob_forward * prob_backward

                              input_candidate, sequence_length_candidate = generate_candidate_input(input, sequence_length, ind, prob_mul, config.search_size, mode=action)
                              prob_candidate_pre = run_epoch(session, mtest_forward, input_candidate, sequence_length_candidate, mode='use')

                              if config.double_LM == True: # default config.double_LM is False
                                    input_candidate_backward, _, _ = reverse_seq(input_candidate, sequence_length_candidate, input_candidate)
                                    prob_candidate_pre += run_epoch(session, mtest_backward, input_candidate_backward, sequence_length_candidate, mode='use')
                                    prob_candidate_pre *= 0.5
                              
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

                              if sim != None: # but config.sim = None(?)
                                    similarity_candidate = similarity_batch(input_candidate, input_original, sta_vec)
                                    prob_candidate *= similarity_candidate

                              prob_candidate_norm = normalize(prob_candidate)
                              prob_candidate_ind = sample_from_candidate(prob_candidate_norm)
                              prob_candidate_prob = prob_candidate[prob_candidate_ind]

                              if input_candidate[prob_candidate_ind][ind + 1] < config.dict_size and (prob_candidate_prob > prob_old_prob * config.threshold or just_acc() == 0):
                                    input = input_candidate[prob_candidate_ind: prob_candidate_ind + 1] # cut at index = prob_candidate_ind
                              
                              pos += 1
                              print ('action: 0', 1, prob_old_prob, prob_candidate_prob, prob_candidate_norm[prob_candidate_ind], similarity_old)
                              if ' '.join(id2sen(input[0])) not in output_p:
                                    outputs.append([' '.join(id2sen(input[0])), prob_old_prob])









# CGMH insertion: line 305-364 (수로)
#word insertion(action:1)




# CGMH deletion: line 367-450 (찬희)
#word deletion(action: 2)