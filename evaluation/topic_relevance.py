from gensim import corpora
from nltk.corpus import wordnet as wn
import pandas as pd
import gensim
import spacy
import nltk

# print(wn.synset('emergent.a.01').lowest_common_hypernyms(wn.synset('issue.n.01')))

def processText(text):
    nlp = spacy.load("en_core_web_md")
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

input_text = ["What can I do if there is emergent game issue?", 
"What can I do if my Flash Player version is too old?",
"Can I control the skills manually in Arena?",
"Why I can't explore a certain stage?",
"Can I challenge the stage I already passed?",
"What's the usage of Mount?",
"Can I get high quality souls with my low quality souls?",
]

sentences = []

for sentence in input_text:
    sentences.append(processText(sentence))

for i in range(len(sentences)):
    sentences[i] = sentences[i].split()

dictionary = corpora.Dictionary(sentences)
corpus = [dictionary.doc2bow(text) for text in sentences]

NUM_TOPICS = 3 # number of topics
ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics = NUM_TOPICS, id2word=dictionary, passes=15)
topics = ldamodel.print_topics()
topic_percentages = ldamodel.get_topics()

# Get the words from ldamodel
words_lst = []
x = ldamodel.show_topics(num_topics=NUM_TOPICS, num_words=5,formatted=False)
topics_words = [(tp[0], [wd[0] for wd in tp[1]]) for tp in x]

# Below Code Prints Only Words 
for topic,words in topics_words:
    lst = []
    word = " ".join(words)
    for x in range(len(words)):
        if wn.synsets(words[x])[0].pos() == 'n':
            lst.append(words[x])
    words_lst.append(lst)

topic_terms = []
for j in range(len(words_lst)):
    tmp = wn.synsets(words_lst[j][0])[0].pos()

    topic_term = wn.synset(words_lst[j][0] + '.' + tmp + '.01').hypernyms()
    for synset in topic_term:
        topic_terms.append(synset.lemmas()[0].name())

    
for i, topic_list in enumerate(ldamodel[corpus]):
    print(i + 1,"th sentence" + "'" + "s topic percentages are: ", topic_list)

for j in range(len(topic_terms)):
    print(j + 1, "th topic: " + topic_terms[j])

    

