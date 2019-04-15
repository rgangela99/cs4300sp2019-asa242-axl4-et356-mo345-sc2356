import json
import re
from nltk.tokenize import TreebankWordTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os
import googleapiclient.discovery
API_KEY = "AIzaSyA2l1Gs_fWKE8-UVWhMgVPmF3Bo2-Sci7U"


tokenizer = TreebankWordTokenizer()
def tokenize(text):
    text= text.lower()
    return tokenizer.tokenize(text)

title_to_text={}
title_to_index={}
link_to_index={}
with open('./data/medium/medium-data-deduped.json') as f:
    data = json.load(f)
i=0
for medium in data:
    title_to_index[medium["title"]]=i
    title_to_text[medium["title"]] = tokenize(medium["text"])
    link_to_index[medium["link"]]=i
    i+=1

yt_index_to_id={}
yt_id_to_text={}
yt_id_to_title={}
with open('./data/reddit/youtube_video_data.json') as f:
    data2 = json.load(f)
i=0
for youtube in data2:
    yt_index_to_id[i]=youtube['id']
    yt_id_to_text[youtube['id']] = tokenize(youtube["snippet"]["description"])
    yt_id_to_title[youtube['id']]=youtube["snippet"]["title"]
    i+=1


n_feats = 5000
doc_by_vocab = np.empty([len(data), n_feats])

def build_vectorizer(max_features, stop_words, max_df=0.8, min_df=10, norm='l2'):
    return TfidfVectorizer(stop_words=stop_words, max_df=max_df, min_df=min_df,max_features=max_features, norm=norm)

tfidf_vec = build_vectorizer(n_feats, "english")
doc_by_vocab = tfidf_vec.fit_transform([d['text'] for d in data]).toarray()
tfidf_vec2 = build_vectorizer(n_feats, "english")
yt_doc_by_vocab = tfidf_vec2.fit_transform([d["snippet"]['description'] for d in data2]).toarray()
index_to_vocab = {i:v for i, v in enumerate(tfidf_vec.get_feature_names())}

def cosine_sim(vec1,doc_by_vocab):
    sims = []
    i=0
    for doc in doc_by_vocab:
        if(np.linalg.norm(vec1)*np.linalg.norm(doc))==0:
            sims.append(0)
        else:
            sims.append(np.dot(vec1,doc)/(np.linalg.norm(vec1)*np.linalg.norm(doc)))
    return sims


def url_to_id(url):
    if '?v=' in url:
        vid_id = url.split('?v=')[1]
        and_idx = vid_id.find('&')

        if and_idx != -1:
            vid_id = vid_id[:and_idx]

        return vid_id
    else:
        return ''

def get_video_info(vids):
    # Disable OAuthlib's HTTPS verification when running locally.
    # *DO NOT* leave this option enabled in production.
#     os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1"

    api_service_name = "youtube"
    api_version = "v3"
    DEVELOPER_KEY = API_KEY

    youtube = googleapiclient.discovery.build(
        api_service_name, api_version, developerKey = DEVELOPER_KEY)

    id_string = ""

    for i in range(len(vids) - 1):
        id_string += vids[i] + ","

    id_string += vids[-1]

    request = youtube.videos().list(
        part="snippet,contentDetails,statistics",
        id=id_string
    )
    response = request.execute()

    return response

def get_single_video(vid_id):
    return get_video_info([vid_id])

def mediumSearch(query):
	vid_id = url_to_id(query)
	api_response = get_single_video(vid_id)
	my_video_info = api_response['items'][0]
	my_title = my_video_info['snippet']['title']
	query_vec = tfidf_vec.transform([my_title]).toarray()
	sims = cosine_sim(query_vec,doc_by_vocab)
	return_arr= []
	for i in range(0,5):
		return_arr.append((data[np.argmax(sims)]["title"],data[np.argmax(sims)]["link"]))
		sims[np.argmax(sims)]=0
	return return_arr

def youtubeSearch(query):
	try:
		article = link_to_index[query]
		text = data[article]["text"]
		query_vec = tfidf_vec2.transform([text]).toarray()
		sims = cosine_sim(query_vec,yt_doc_by_vocab)
		return_arr= []
		for i in range(0,5):
			return_arr.append((yt_id_to_title[yt_index_to_id[np.argmax(sims)]],"https://www.youtube.com/watch?v="+yt_index_to_id[np.argmax(sims)]))
			sims[np.argmax(sims)]=0
		return return_arr
	except:

		return [("Exception","")]




def getLink(query):
    if(query == ""):
        return 0
    else:
        return mediumSearch(query)
