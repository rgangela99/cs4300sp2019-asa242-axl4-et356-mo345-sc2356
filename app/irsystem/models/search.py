import json
import re
import pickle
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
#for SVD
from sklearn.decomposition import TruncatedSVD
#for Sentiment Analysis
#import nltk
#nltk.download('vader_lexicon')
#from nltk.sentiment.vader import SentimentIntensityAnalyzer
#sid = SentimentIntensityAnalyzer()


#general purpose tokenizer for text input
tokenizer = TreebankWordTokenizer()
def tokenize(text):
    text= text.lower()
    return tokenizer.tokenize(text)


def claps_to_nums(claps):
    if claps == 0:
        return 0
    num=claps.split()[0]
    if "K" in num:
        num=num[:-1]
        num=float(num)*1000
    else:
        num=float(num)
    return num

#building data arrays for Medium article text and YouTube video plus tags
#for those that have tags
med_text_tag = []
yt_title_tag = []

#dictionary for referencing the Medium article data set
with open('./data/medium/medium-data.pickle', 'rb') as f:
    medium_ind_to_art_info = pickle.load(f)

med_data_len = len(medium_ind_to_art_info.keys())

#dictionaries for referencing YouTube video data set
with open('./data/reddit/youtube-index-id.pickle', 'rb') as f:
    yt_index_to_id = pickle.load(f)

with open('./data/reddit/youtube-vid-info.pickle', 'rb') as f:
    yt_id_to_vid_info = pickle.load(f)

yt_data_len = len(yt_index_to_id.keys())

#matrices
with open('./data/reddit/youtube_video_lengths.pickle', 'rb') as f:
    yt_id_to_length = pickle.load(f)

with open('./data/medium/medium-matrix.pickle', 'rb') as f:
    medium_articles_by_vocab = pickle.load(f)

with open('./data/reddit/youtube-matrix.pickle', 'rb') as f:
    yt_vids_by_vocab = pickle.load(f)

#vectorizer
with open('./data/vectorizer.pickle', 'rb') as f:
    tfidf_vec = pickle.load(f)

#SVD
with open('./data/SVD-med-model.pickle', 'rb') as f:
    svd_med = pickle.load(f)
with open('./data/SVD-yt-model.pickle', 'rb') as f:
    svd_yt = pickle.load(f)
with open('./data/SVD-med-docs.pickle', 'rb') as f:
    svd_med_docs = pickle.load(f)
with open('./data/SVD-yt-docs.pickle', 'rb') as f:
    svd_yt_docs = pickle.load(f)

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

def vid_url_to_title(vid_url):
    return get_single_video(url_to_id(vid_url))['items'][0]['snippet']['title']

def art_url_to_title(art_url):
    data = requests.get(art_url)
    soup = BeautifulSoup(data.content, 'html.parser')
    title = soup.findAll('title')[0]
    title = title.get_text()
    return title

def youtubeKeywords(keywords):
    key_calc_arr=np.zeros(len(yt_index_to_id))
    i=0
    if(keywords!=""):
        keyword_arr = keywords.split(",")
    else:
        keyword_arr = [keywords]
    for vid in yt_id_to_vid_info.keys():
        if 'tags' in yt_id_to_vid_info[vid].keys():
            key_calc_arr[i]=len(yt_id_to_vid_info[vid]["tags"] & set(keyword_arr))
        i+=1

    return (key_calc_arr)

def mediumKeywords(keywords):
    key_calc_arr=np.zeros(len(medium_ind_to_art_info))
    i=0
    if(keywords!=""):
        keyword_arr = keywords.split(",")
    else:
        keyword_arr = [keywords]
    for art in medium_ind_to_art_info.values():
        if "tags" in art.keys():
            key_calc_arr[i]=len(art["tags"] & set(keyword_arr))
        i+=1
    return (key_calc_arr)

def youtubeComments():
    comment_score_arr = np.zeros(len(yt_index_to_id))
    i=0
    for vid_info in yt_id_to_vid_info.values():
        has_comments = ("comment_toks" in vid_info.keys())
        has_tags = ("tags" in vid_info.keys())
        if (has_comments and has_tags):
            comments = vid_info["comment_toks"]
            tags = vid_info["tags"]
            comment_score_arr[i] = len(comments & tags)
        i+=1
    return comment_score_arr


def mediumComments():
    comment_score_arr = np.zeros(len(medium_ind_to_art_info))
    i=0
    for article in medium_ind_to_art_info.values():
        has_comments = ("comment_toks" in article.keys())
        has_tags = ("tags" in article.keys())
        if (has_comments and has_tags):
            comments = article["comment_toks"]
            tags = article["tags"]
            comment_score_arr[i] = len(comments & tags)
        i+=1
    return comment_score_arr

med_comment_weight = 0.01
yt_comment_weight = 0.01
keyword_weight = 0.1
med_sentiment_weight = 0.01
yt_sentiment_weight = 0.01
med_sentiment_cap = 0.03
yt_sentiment_cap = 0.03

#YouTube video comment scores
yt_comment_scores = yt_comment_weight*youtubeComments()
#Medium article comment scores
med_comment_scores = med_comment_weight*mediumComments()

yt_sentiment_scores = []

for k in yt_id_to_vid_info.keys():
    curr_score = 0
    if 'sentiments' in yt_id_to_vid_info[k].keys():
        for comm_sent in yt_id_to_vid_info[k]['sentiments']:
            curr_score += yt_sentiment_weight * comm_sent['compound']
    
    yt_sentiment_scores.append(max(curr_score, yt_sentiment_cap))

medium_sentiment_scores = []

for k in medium_ind_to_art_info.keys():
    curr_score = 0
    if 'sentiments' in medium_ind_to_art_info[k].keys():
        for comm_sent in medium_ind_to_art_info[k]['sentiments']:
            curr_score += med_sentiment_weight * comm_sent['compound']  

    medium_sentiment_scores.append(max(curr_score, med_sentiment_cap))

#search function from YouTube video to Medium article
def mediumSearch(query,keywords,max_time):
    num_results = 10
    vid_id = url_to_id(query)
    api_response = get_single_video(vid_id)
    my_video_info = api_response['items'][0]
    my_title = my_video_info['snippet']['title']
    tags=" "
    if 'tags' in my_video_info["snippet"].keys():
        for tag in my_video_info["snippet"]["tags"]:
            tags=tag+" "

    query_vec = tfidf_vec.transform([my_title + tags]).toarray()

    return_arr = []
    
    svd_query = svd_med.transform(query_vec)
    weighted_keywords = keyword_weight*mediumKeywords(keywords)
    sims = np.array(cosine_sim(svd_query, svd_med_docs)).flatten()+weighted_keywords+med_comment_scores+medium_sentiment_scores
    sort_idx = np.flip(np.argsort(sims))
    id_arr = []
    
    num_found = 0
    n = len(sort_idx)
    for i in range(0,n):
        article = medium_ind_to_art_info[sort_idx[i]]
        if article["reading_time"] <= max_time:
            return_arr.append((article["title"]+" "+str(sims[sort_idx[i]]), article["link"], article["comments"][0] if ("comments" in article.keys()) else "", article["claps"], article["reading_time"]))
            id_arr.append(sort_idx[i])
        if num_found == num_results:
            break

    return return_arr

#search function from Medium article to YouTube video
def youtubeSearch(query,keywords,max_time):
    num_results = 10
    data = requests.get(query)
    soup = BeautifulSoup(data.content, 'html.parser')
    paras = soup.findAll('p')
    text = ''
    nxt_line = '\n'
    for para in paras:
        text += unicodedata.normalize('NFKD',
                                        para.get_text()) + nxt_line
    
    title = soup.findAll('title')[0]
    title = title.get_text()

    tags = soup.findAll('ul', 'tags')
    if tags:
        tags_list = {t.findAll('a')[0].get_text().lower() for t in tags[0]}
        tags=" "
        for tag in tags_list:
        	tags=tag+" "
    else:
        tags=""

    query_vec = tfidf_vec.transform([text+" "+title + tags]).toarray()
    return_arr= []
    svd_query = svd_yt.transform(query_vec)
    weighted_keywords = keyword_weight*youtubeKeywords(keywords)
    sims = np.array(cosine_sim(svd_query,svd_yt_docs)).flatten()+weighted_keywords+yt_comment_scores+yt_sentiment_scores
    sort_idx = np.flip(np.argsort(sims))
    id_arr = []

    num_found = 0
    n = len(sort_idx)
    for i in range(0,n):
        curr_id = yt_index_to_id[sort_idx[i]]
        if yt_id_to_length[curr_id] <= max_time:
            return_arr.append((yt_id_to_vid_info[curr_id]["title"]+" "+str(sims[sort_idx[i]]),"https://www.youtube.com/watch?v="+curr_id, yt_id_to_vid_info[curr_id]["comments"][0] if ("comments" in yt_id_to_vid_info[curr_id].keys()) else "", yt_id_to_vid_info[curr_id]["likes"], round(yt_id_to_length[curr_id])))
            id_arr.append(curr_id)
            num_found += 1
        if num_found == num_results:
            break


    return return_arr
    
def getLink(query):
    if(query == ""):
        return 0
    else:
        return mediumSearch(query)
