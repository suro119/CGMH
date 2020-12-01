#word deletion(action: 2)
if action == 2  and ind < sequence_length[0] - 1:
if sequence_length[0] <= 2:
# skip word
action = 3
else:
prob_old = run_epoch(session, mtest_forward, input, sequence_length, mode='use')
if config.double_LM == True:
  # reverse the sequence
  input_backward, _, _ = reverse_seq(input, sequence_length, input)
  prob_old = (prob_old + run_epoch(session, mtest_backward, input_backward, sequence_length, mode='use')) * 0.5

tem = 1
for j in range(sequence_length[0] - 1):
  tem *= prob_old[0][j][input[0][j+1]]
tem *= prob_old[0][j+1][config.dict_size+1]
prob_old_prob = tem

# refer to 218 fro sta_vec
if sim != None:
  similarity_old = similarity(input[0], input_original, sta_vec)
  prob_old_prob = prob_old_prob * similarity_old
else:
  similarity_old = -1

input_candidate, sequence_length_candidate = generate_candidate_input(input, sequence_length, ind, None, config.search_size, mode=2)
prob_new = run_epoch(session, mtest_forward, input_candidate, sequence_length_candidate, mode='use')

# reset tem
tem = 1
for j in range(sequence_length_candidate[0]-1):
  tem *= prob_new[0][j][input_candidate[0][j+1]]
tem *= prob_new[0][j+1][config.dict_size+1]
prob_new_prob = tem

if sim != None:
  similarity_new = similarity_batch(input_candidate, input_original, sta_vec)
  prob_new_prob = prob_new_prob * similarity_new

input_forward, input_backward, sequence_length_forward, sequence_length_backward = cut_from_point(input, sequence_length, ind, mode=0)
prob_forward = run_epoch(session, mtest_forward, input_forward, sequence_length_forward, mode='use')[0, ind % (sequence_length[0]-1),:]
prob_backward = run_epoch(session, mtest_backward, input_backward, sequence_length_backward, mode='use')[0, sequence_length[0] - 1 - ind%(sequence_length[0]-1),:]
prob_mul = (prob_forward * prob_backward)
input_candidate, sequence_length_candidate = generate_candidate_input(input, sequence_length, ind, prob_mul, config.search_size, mode=0)
prob_candidate_pre = run_epoch(session, mtest_forward, input_candidate, sequence_length_candidate, mode='use')

if config.double_LM == True:
  input_candidate_backward, _, _ = reverse_seq(input_candidate, sequence_length_candidate, input_candidate)
  prob_candidate_pre = (prob_candidate_pre+run_epoch(session, mtest_backward, input_candidate_backward, sequence_length_candidate, mode='use'))*0.5

prob_candidate = []
for i in range(config.search_size):
  # reset tem
  tem = 1
  for j in range(sequence_length[0]-1):
    tem *= prob_candidate_pre[i][j][input_candidate[i][j+1]]
  tem *= prob_candidate_pre[i][j+1][config.dict_size+1]
  prob_candidate.append(tem)
prob_candidate = np.array(prob_candidate)

if sim != None:
  similarity_candidate = similarity_batch(input_candidate, input_original,sta_vec)
  prob_candidate = prob_candidate * similarity_candidate

#alpha is acceptance ratio of current proposal
prob_candidate_norm = normalize(prob_candidate)
if input[0] in input_candidate:
  for candidate_ind in range(len(input_candidate)):
    if input[0] in input_candidate[candidate_ind: candidate_ind+1]:
      break
    pass
  # calculate the acceptance rate
  xprime = prob_candidate_norm[candidate_ind] * prob_new_prob * config.action_prob[1]
  x_t = config.action_prob[2] * prob_old_prob
  alpha = min(xprime/x_t, 1)
else:
  pass
  alpha = 0

print('action:2', alpha, prob_old_prob, prob_new_prob, prob_candidate_norm[candidate_ind], similarity_old)
if ' '.join(id2sen(input[0])) not in output_p:
  outputs.append([' '.join(id2sen(input[0])), prob_old_prob])

if choose_action([alpha, 1 - alpha]) == 0 and (prob_new_prob > prob_old_prob * config.threshold or just_acc() == 0):
  input = np.concatenate([input[:,:ind+1], input[:,ind+2:], input[:,:1] * 0 + config.dict_size + 1], axis=1)
  # deleted
  sequence_length -= 1
  pos += 0
  del(sta_vec[ind])
  sta_vec.append(0)
else:
  action = 3