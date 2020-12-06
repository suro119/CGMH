import spacy

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
        result.append(token.lemma_)
    return " ".join(result)


def compareSents(sent1, sent2):
    nlp = spacy.load("en_core_web_lg")

    # Sentence level comparison
    tokens1 = nlp(sent1)
    tokens2 = nlp(sent2)
    sent_similarity = tokens1.similarity(tokens2)
    print("1. Sentence level comparison:", sent_similarity, "")

    # Word level comparison
    sent1_f = processText(sent1)
    sent2_f = processText(sent2)
    print("* Processed Sent1: [", sent1_f, "] / Sent2: [", sent2_f, "]")
    tokens1 = nlp(sent1_f)
    tokens2 = nlp(sent2_f)

    processed_sent_similarity = tokens1.similarity(tokens2)
    print("2. Processed sentence level comparison:", processed_sent_similarity, "")

    length = 0
    score = 0

    for token1 in tokens1:
        for token2 in tokens2:
            if (token1 and token1.vector_norm and token2 and token2.vector_norm):
            # print("*", token1.text, token2.text, token1.similarity(token2))
                length += 1
                score += token1.similarity(token2)
    word_similarity = score/length

    print("3. Word level comparison:", word_similarity, "")
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
inp = ["How do I make the planes fly?", 
"What can I do if there is emergent game issue?", 
"What can I do if my Flash Player version is too old?",
"What can I do if the game stuck on loading page?",
"What can I do if the Flash Player crashed?",
"What can I do if I can't login?",
"How could I get the chance to play the game?",
"Can I get high quality souls with my low quality souls?",
"How many qualities of Hero Souls there are?",
"What's the usage of Hero Soul?",
"Why some players get better reward from Conquest?",
"Is all the mining area the same?",
"Why can't I attack some players' mine?",
"What's the benefit if a guild occupied a mine area?"]
'''

inp = ["Where is the nearest convenience store from the campus?", 
"What should I have for dinner, inside or outside?", 
"Can you recommend good restaurants near west gate?",
"Where do you think is better, lotteria or Kaimaru?"]

calcSimilarityScore(inp)
