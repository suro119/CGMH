import json, sys

with open(sys.argv[1]) as f:
    messages = json.load(f)

#print(messages['messages'])
messages = messages['messages']

# for msg in messages:
#     print(msg)

# print()

botMessages = list(map(lambda x: x['content'], filter(lambda x: x['sender_name'] == 'Pandorabots', sorted(messages, key=lambda x: x['timestamp_ms']))))
# for msg in botsMessages:
#     print(msg)

# print()
'''
indices = [i for i, x in enumerate(botMessages) if x == "Is that a smiley face?"]

'''
indices = [15, 30, 45, 60, 75]
indices.insert(0, 0)
indices.append(len(botMessages))

print(indices)

cutMessages = []
for i in range(len(indices)-1):
    cutMessages.append(botMessages[indices[i]:indices[i+1]])

print(len(botMessages))


with open('data2.txt', 'w') as outfile:
    json.dump(cutMessages, outfile)