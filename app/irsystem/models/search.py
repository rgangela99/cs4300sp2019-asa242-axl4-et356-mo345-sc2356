import json
import re
from nltk.tokenize import TreebankWordTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os
#for YouTube video scraping
import googleapiclient.discovery
import urllib3
from bs4 import BeautifulSoup
import requests
import unicodedata
API_KEY = "AIzaSyA2l1Gs_fWKE8-UVWhMgVPmF3Bo2-Sci7U"

#general purpose tokenizer for text input
tokenizer = TreebankWordTokenizer()
def tokenize(text):
    text= text.lower()
    return tokenizer.tokenize(text)

#building data array of both article text and video description text 
#to train the vectorizer
data = []

#dictionaries for referencing the Medium article data set
title_to_text={}
title_to_index={}
link_to_index={}
with open('./data/medium/medium-data-deduped.json') as f:
    medium_data = json.load(f)
i=0
for article in medium_data:
    title_to_index[article["title"]]=i
    title_to_text[article["title"]] = tokenize(article["text"])
    link_to_index[article["link"]]=i
    data.append(article["text"])
    i+=1

#dictionaries for referencing the YouTube videos data set
yt_index_to_id={}
yt_id_to_text={}
yt_id_to_title={}
yt_id_to_likes={}
with open('./data/reddit/youtube_video_data.json') as f:
    yt_data = json.load(f)
i=0
for youtube in yt_data:
    yt_index_to_id[i]=youtube['id']
    yt_id_to_text[youtube['id']] = tokenize(youtube["snippet"]["description"])
    yt_id_to_title[youtube['id']]=youtube["snippet"]["title"]
    yt_id_to_likes[youtube['id']]=0 
    if 'statistics' in youtube.keys():
        if 'likeCount' in youtube['statistics'].keys():
            yt_id_to_likes[youtube['id']]=int(youtube['statistics']['likeCount'])
    data.append(youtube["snippet"]["description"])
    i+=1

#maximum number of features to train the vectorizer
n_feats = 5000
medium_articles_by_vocab = np.empty([len(medium_data), n_feats])
yt_vids_by_vocab = np.empty([len(yt_data), n_feats])
# doc_by_vocab = np.empty([len(data), n_feats])

def build_vectorizer(max_features, stop_words, max_df=0.8, min_df=10, norm='l2'):
    return TfidfVectorizer(stop_words=stop_words, max_df=max_df, min_df=min_df,max_features=max_features, norm=norm)

#building vectorizer to train
tfidf_vec = build_vectorizer(n_feats, "english")
tfidf_vec.fit(d for d in data)
medium_articles_by_vocab = tfidf_vec.transform(art["text"] for art in medium_data).toarray()
yt_vids_by_vocab = tfidf_vec.transform(vid["snippet"]["description"] for vid in yt_data).toarray()
# doc_by_vocab = tfidf_vec.fit_transform([d['text'] for d in data]).toarray()
# tfidf_vec2 = build_vectorizer(n_feats, "english")
# yt_doc_by_vocab = tfidf_vec2.fit_transform([d["snippet"]['description'] for d in data2]).toarray()
index_to_vocab = {i:v for i, v in enumerate(tfidf_vec.get_feature_names())}

#returns list of cosine similarities of query vector with every document in provided
#tf-idf matrix [doc_by_vocab]
def cosine_sim(vec1,doc_by_vocab):
    sims = []
    i=0
    for doc in doc_by_vocab:
        if(np.linalg.norm(vec1)*np.linalg.norm(doc))==0:
            sims.append(0)
        else:
            sims.append(np.dot(vec1,doc)/(np.linalg.norm(vec1)*np.linalg.norm(doc)))
    return sims

#YouTube video scraping
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

def claps_to_nums(claps):
	num=claps.split()[0]
	if "K" in num:
		num=num[:-1]
		num=float(num)*1000
	else:
		num=float(num)
	return num

#search function from YouTube video to Medium article
def mediumSearch(query):
    vid_id = url_to_id(query)
    api_response = get_single_video(vid_id)
    my_video_info = api_response['items'][0]
    my_title = my_video_info['snippet']['title']
    query_vec = tfidf_vec.transform([my_title]).toarray()
    sims = cosine_sim(query_vec,medium_articles_by_vocab)
    return_arr = []
    sort_idx = np.argsort(sims)
    for i in range(0,5):
        # article = medium_data[sort_idx[i]]
        # return_arr.append((article["title"], article["link"]))
        # return_arr.append((data[np.argmax(sims)]["title"],data[np.argmax(sims)]["link"]))
        # sims[np.argmax(sims)]=0
        return_arr.append((medium_data[np.argmax(sims)]["title"],medium_data[np.argmax(sims)]["link"]))
        sims[np.argmax(sims)]=0
    clap_arr = []
    for j in range(0,5):
    	art_index = title_to_index[return_arr[j][0]]
    	claps=medium_data[art_index]["claps"]
    	claps_to_nums(claps)
    	clap_arr.append(claps_to_nums(claps))
    clap_return_arr=[]
    for k in range(0,5):
    	clap_return_arr.append(return_arr[np.argmax(clap_arr)])
    	clap_arr[np.argmax(clap_arr)]=0
    return clap_return_arr

#search function from Medium article to YouTube video
def youtubeSearch(query):
    try:
        data = requests.get(query)
        soup = BeautifulSoup(data.content, 'html.parser')
        paras = soup.findAll('p')
        text = ''
        nxt_line = '\n'
        for para in paras:
            text += unicodedata.normalize('NFKD',
                                            para.get_text()) + nxt_line
        query_vec = tfidf_vec.transform([text]).toarray()
        sims = cosine_sim(query_vec,yt_vids_by_vocab)
        return_arr= []
        sort_idx = np.argsort(sims)
        
        for i in range(0,5):
            return_arr.append((yt_id_to_title[yt_index_to_id[np.argmax(sims)]],"https://www.youtube.com/watch?v="+yt_index_to_id[np.argmax(sims)]))
            id_arr.append(yt_index_to_id[np.argmax(sims)])
            sims[np.argmax(sims)]=0
        
        like_arr = [yt_id_to_likes[i] for i in in_arr]
        like_return_arr=[]
        for k in range(0,5):
            like_return_arr.append(return_arr[np.argmax(like_arr)])
            like_arr[np.argmax(like_arr)]=0
            
        return like_return_arr
    except Exception as e:
        print(e)
        return [("This is not a recognized Medium article link","")]

def getLink(query):
    if(query == ""):
        return 0
    else:
        return mediumSearch(query)
