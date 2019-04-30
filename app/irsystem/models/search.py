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

#weights
med_comment_weight = 0.01
yt_comment_weight = 0.01
keyword_weight = 0.3
med_sentiment_weight = 0.001
yt_sentiment_weight = 0.001
yt_likes_weight = 0.1
med_claps_weight = 0.1

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

#array for YouTube video likes
with open('./data/likes-array.pickle', 'rb') as f:
    likes_arr = pickle.load(f)
likes_arr = yt_likes_weight*likes_arr

#array for Medium article claps
with open('./data/claps-array.pickle', 'rb') as f:
    claps_arr = pickle.load(f)
claps_arr = med_claps_weight*claps_arr

#array for Medium article sentiment scores
with open('./data/med-sentiment.pickle', 'rb') as f:
    medium_sentiment_scores = pickle.load(f)
medium_sentiment_scores = med_sentiment_weight*medium_sentiment_scores

#array for YouTube video sentiment scores
with open('./data/yt-sentiment.pickle', 'rb') as f:
    yt_sentiment_scores = pickle.load(f)
yt_sentiment_scores = yt_sentiment_weight*yt_sentiment_scores

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

def youtubeComments(tag_set):
    comment_score_arr = np.zeros(yt_data_len)
    i=0
    for vid_info in yt_id_to_vid_info.values():
        if ("comment_toks" in vid_info.keys()):
            comments = vid_info["comment_toks"]
            comment_score_arr = len(comments & tag_set)
        i+=1
    return comment_score_arr

def mediumComments(tag_set):
    comment_score_arr = np.zeros(med_data_len)
    i=0
    for article in medium_ind_to_art_info.values():
        if ("comment_toks" in article.keys()):
            comments = article["comment_toks"]
            comment_score_arr[i] = len(comments & tag_set)
        i+=1
    return comment_score_arr

#search function from YouTube video to Medium article
def mediumSearch(query,keywords,max_time):
    num_results = 10
    vid_id = url_to_id(query)
    api_response = get_single_video(vid_id)
    my_video_info = api_response['items'][0]
    my_title = my_video_info['snippet']['title']
    tags=" "
    tag_set = set()
    if 'tags' in my_video_info["snippet"].keys():
        for tag in my_video_info["snippet"]["tags"]:
            tags=tag+" "
        tag_set.update(tokenize(tags))
    query_vec = tfidf_vec.transform([my_title + tags]).toarray()

    return_arr = []
    
    svd_query = svd_med.transform(query_vec)
    weighted_keywords = keyword_weight*mediumKeywords(keywords)
    med_comment_scores = med_comment_weight*mediumComments(tag_set)
    cos_sims = np.array(cosine_sim(svd_query, svd_med_docs)).flatten()
    sims = cos_sims+weighted_keywords+med_comment_scores+medium_sentiment_scores+claps_arr
    sort_idx = np.flip(np.argsort(sims))
    id_arr = []
    
    num_found = 0
    n = len(sort_idx)
    for i in range(0,n):
        article = medium_ind_to_art_info[sort_idx[i]]
        s_i = sort_idx[i]
        if article["reading_time"] <= max_time:
            scores_tuple = (cos_sims[s_i], weighted_keywords[s_i], med_comment_scores[s_i], medium_sentiment_scores[s_i], claps_arr[s_i])
            return_arr.append((article["title"][:min(len(article["title"]),77)]+" "+str(sims[sort_idx[i]]), article["link"], ', '.join(article["tags"]), article["claps"], article["reading_time"]),scores_tuple)
            id_arr.append(sort_idx[i])
            num_found+=1
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
        tag_set = set()
        for tag in tags_list:
            tags=tag+" "
        tag_set.update(tokenize(tags))
    else:
        tags=""

    query_vec = tfidf_vec.transform([text+" "+title + tags]).toarray()
    return_arr= []
    svd_query = svd_yt.transform(query_vec)
    weighted_keywords = keyword_weight*youtubeKeywords(keywords)
    yt_comment_scores = yt_comment_weight*youtubeComments(tag_set)
    cos_sims = np.array(cosine_sim(svd_query,svd_yt_docs)).flatten()
    sims = cos_sims+weighted_keywords+yt_comment_scores+yt_sentiment_scores+likes_arr
    sort_idx = np.flip(np.argsort(sims))
    id_arr = []

    num_found = 0
    n = len(sort_idx)
    for i in range(0,n):
        curr_id = yt_index_to_id[sort_idx[i]]
        s_i = sort_idx[i]
        if yt_id_to_length[curr_id] <= max_time:
            scores_tuple = (cos_sims[s_i], weighted_keywords[s_i], yt_comment_scores[s_i], yt_sentiment_scores[s_i], likes_arr[s_i])
            return_arr.append((yt_id_to_vid_info[curr_id]["title"][:min(len(yt_id_to_vid_info[curr_id]["title"]),77)]+" "+str(sims[sort_idx[i]]),"https://www.youtube.com/watch?v="+curr_id, ', '.join(yt_id_to_vid_info[curr_id]["tags"]) if "tags" in yt_id_to_vid_info[curr_id] else "", yt_id_to_vid_info[curr_id]["likes"], round(yt_id_to_length[curr_id])),scores_tuple)
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
