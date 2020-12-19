import spacy
import json
import sys

def processText(text):
    nlp = spacy.load("en_core_web_lg")
    doc = nlp(text.lower())
    result = []
    for token in doc:
        if token.text in nlp.Defaults.stop_words:
            continue
        if token.is_punct:
            continue
        if token.lemma_ == '-PRON-':
            continue
        if 'music' in token.text:
            continue
        result.append(token.lemma_)
    return " ".join(result)


def compareSents(sent1, sent2):
    nlp = spacy.load("en_core_web_lg")

    # Sentence level comparison
    tokens1 = nlp(sent1)
    tokens2 = nlp(sent2)
    sent_similarity = 0
    #sent_similarity = tokens1.similarity(tokens2)
    #print("1. Sentence level comparison:", sent_similarity, "")

    # Word level comparison
    sent1_f = processText(sent1)
    sent2_f = processText(sent2)
    print("* Processed Sent1: [", sent1_f, "] / Sent2: [", sent2_f, "]")
    tokens1 = nlp(sent1_f)
    tokens2 = nlp(sent2_f)

    processed_sent_similarity = tokens1.similarity(tokens2)
    print("2. Processed sentence level comparison:", processed_sent_similarity, "")

    word_similarity = 0
    '''
    length = 0
    score = 0
    word_similarity = 0
    for token1 in tokens1:
        for token2 in tokens2:
            if (token1 and token1.vector_norm and token2 and token2.vector_norm):
            # print("*", token1.text, token2.text, token1.similarity(token2))
                length += 1
                score += token1.similarity(token2)
    if (length > 0):
        word_similarity = score/length

    print("3. Word level comparison:", word_similarity, "")
    '''
    return sent_similarity, processed_sent_similarity, word_similarity
    


def calcSimilarityScore(inp):
    print("\nBEGINNING -")
    print("Total number of input sentences:", len(inp))
    out = []

    for i in range(len(inp)):
        for j in range(len(inp)-i):
            if (not i == j+i):
                print("----------------------\nComparing\n#", i, ": [", inp[i], "] AND #", j+i, ": [", inp[j+i], "]\n")
                sent_similarity, processed_sent_similarity, word_similarity = compareSents(inp[i], inp[j+i])
                out.append({"sent1": i, "sent2": j+i, "sent_score": sent_similarity, "p_sent_score": processed_sent_similarity, "word_score": word_similarity})

    print("----------------------\nList of comparisons")
    for i in out:
        #print(i)
        print('#{0[sent1]} vs. #{0[sent2]} | sent: {0[sent_score]:.4f} | p-sent: {0[p_sent_score]:.4f} | word: {0[word_score]:.4f}'.format(i))

    avg_sent_score = sum(list(map(lambda x: x["sent_score"], out)))/len(out)
    avg_p_sent_score = sum(list(map(lambda x: x["p_sent_score"], out)))/len(out)
    avg_word_score = sum(list(map(lambda x: x["word_score"], out)))/len(out)

    print("----------------------")
    print("FINAL RESULTS -")
    print("Length of list:", len(out))
    print("Average sent_score:", round(avg_sent_score, 4))
    print("Average p_sent_score:", round(avg_p_sent_score, 4))
    print("Average word_score:", round(avg_word_score, 4))

    return avg_sent_score, avg_p_sent_score, avg_word_score

'''
inp = [
"What music do you prefer?",
"Do you know Rachmaninoff piano concerto no.2?",
"Do you like classical music?",
"How many times did you listen to Gangnam Style?",
"What is the pitch of this song?",
"What is your favorite music?",
"Can I listen to music on YouTube?",
"What’s your KakaoTalk profile music?",
"Do you often listen to music?",
"Are you part of any musical groups?",
"What genre of music do you prefer?",
"Do you like Brahms?",
"Do you like Vitas?",
"Have you listened to Eminem?",
"What is your favorite idol music?"
]
'''

# print(sys.argv[1], sys.argv[2])
print("RESULTS from", sys.argv[1], 'to', sys.argv[2])


with open('data.txt') as f:
    messages = json.load(f)


for msglist in messages[int(sys.argv[1]): int(sys.argv[2])+1]:
    sent_score, p_sent_score, word_score = calcSimilarityScore(msglist[:])
    result = {'index': messages.index(msglist), 'p_sent_score': p_sent_score}
    with open('result.txt', 'a') as outfile:
        json.dump(result, outfile)
