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





# CGMH insertion: line 305-364 (수로)
#word insertion(action:1)




# CGMH deletion: line 367-450 (찬희)
#word deletion(action: 2)