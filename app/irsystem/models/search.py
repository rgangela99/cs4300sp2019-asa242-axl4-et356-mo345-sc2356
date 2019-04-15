import json
import re
from nltk.tokenize import TreebankWordTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

tokenizer = TreebankWordTokenizer()
def tokenize(text):
    text= text.lower()
    return tokenizer.tokenize(text)

title_to_text={}
title_to_index={}
with open('./data/medium/medium-data-small.json') as f:
    data = json.load(f)
i=0
for medium in data:
    title_to_index[medium["title"]]=i
    title_to_text[medium["title"]] = tokenize(medium["text"])
    i+=1

n_feats = 5000
doc_by_vocab = np.empty([len(data), n_feats])

def build_vectorizer(max_features, stop_words, max_df=0.8, min_df=10, norm='l2'):
    return TfidfVectorizer(stop_words=stop_words, max_df=max_df, min_df=min_df,max_features=max_features, norm=norm)

tfidf_vec = build_vectorizer(n_feats, "english")
doc_by_vocab = tfidf_vec.fit_transform([d['text'] for d in data]).toarray()
index_to_vocab = {i:v for i, v in enumerate(tfidf_vec.get_feature_names())}

def mediumSearch(query):
	query_vec = tfidf_vec.transform([query]).toarray()
	sims = cosine_sim(query_vec,doc_by_vocab)
	return_arr= []
	for i in range(0,5):
		return_arr.append(data[np.argmax(sims)]["title"])
		sims[np.argmax(sims)]=0
	return return_arr


def cosine_sim(vec1,doc_by_vocab):
    sims = []
    i=0
    for doc in doc_by_vocab:
        if(np.linalg.norm(vec1)*np.linalg.norm(doc))==0:
            sims.append(0)
        else:
            sims.append(np.dot(vec1,doc)/(np.linalg.norm(vec1)*np.linalg.norm(doc)))
    return sims

def getLink(query):    
    if(query == ""):
        return 0
    else:
        return mediumSearch(query)



