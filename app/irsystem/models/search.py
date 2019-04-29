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

#general purpose tokenizer for text input
tokenizer = TreebankWordTokenizer()
def tokenize(text):
    text= text.lower()
    return tokenizer.tokenize(text)

#building data arrays for Medium article text and YouTube video plus tags
#for those that have tags
med_text_tag = []
yt_title_tag = []

#dictionaries for referencing the Medium article data set
title_to_text={}
title_to_index={}
link_to_index={}
title_to_tags={}
with open('./data/medium/deduped-medium-comments-list.json') as f:
    medium_data = json.load(f)
i=0
for article in medium_data:
    title_to_index[article["title"]]=i
    title_to_text[article["title"]] = tokenize(article["text"])
    link_to_index[article["link"]]=i
    tags=" "
    if "tags" in article.keys():
        for tag in article["tags"]:
            tags=tag+" "
        title_to_tags[article["title"]] = tags
    art_text_tag = article["text"]+tags
    med_text_tag.append(art_text_tag)
    i+=1

with open('./data/reddit/youtube_comment_data.json') as f:
    yt_comment_data = json.load(f)

with open('./data/reddit/youtube_video_lengths.pickle', 'rb') as f:
    yt_id_to_length = pickle.load(f)

#dictionaries for referencing the YouTube videos data set
yt_index_to_id={}
yt_id_to_text={}
yt_id_to_title={}
yt_id_to_likes={}
yt_id_to_comment={}
yt_id_to_tags={}
with open('./data/reddit/youtube_video_data.json') as f:
    yt_data = json.load(f)

#we can concatenate all relevant comments into a single string
for vid_comments in yt_comment_data:
    concatenated_top_comments = ""
    for comment in vid_comments["text_likes"]:
        concatenated_top_comments += comment[0]
    yt_id_to_comment[vid_comments["id"]] = concatenated_top_comments

i=0
for youtube in yt_data:
    yt_index_to_id[i]=youtube['id']
    yt_id_to_text[youtube['id']] = tokenize(youtube["snippet"]["description"])
    yt_id_to_title[youtube['id']]=youtube["snippet"]["title"]
    yt_id_to_likes[youtube['id']]=0
    if 'statistics' in youtube.keys():
        if 'likeCount' in youtube['statistics'].keys():
            yt_id_to_likes[youtube['id']]=int(youtube['statistics']['likeCount'])
    if youtube['id'] not in yt_id_to_comment.keys():
        yt_id_to_comment[youtube['id']]=""
    tags=" "
    if 'tags' in youtube["snippet"].keys():
        for tag in youtube["snippet"]["tags"]:
            tags=tag+" "
        yt_id_to_tags[youtube['id']]=tags
    vid_title_tag = youtube["snippet"]["title"]+tags
    yt_title_tag.append(vid_title_tag)
    i+=1

#data array of both article text and video description text
#to train the vectorizer
data = med_text_tag + yt_title_tag

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
medium_articles_by_vocab = tfidf_vec.transform(art for art in med_text_tag).toarray()
yt_vids_by_vocab = tfidf_vec.transform(vid for vid in yt_title_tag).toarray()
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



def SVD(tf_idf_matrix, query, k_val):
#     svd_matrix = (tf_idf_matrix).transpose()
#     u, sigma, docs_compressed = svds(svd_matrix, k=k_val)
#     docs_compressed = docs_compressed.transpose()
#     docs_compressed = normalize(docs_compressed, axis = 1)
#     return (np.array(u),np.array(sigma),np.array(docs_compressed))
    svd = TruncatedSVD(n_components=k_val)
    return (svd.fit_transform(tf_idf_matrix), svd.transform(query)) 

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

def youtubeKeywords(keywords):
    key_calc_arr=np.zeros(len(yt_index_to_id))
    i=0
    if(keywords!=""):
        keyword_arr = keywords.split(",")
    else:
        keyword_arr = [keywords]
    for youtube in yt_data:
        if 'tags' in youtube["snippet"].keys():
            key_calc_arr[i]=len(set(youtube["snippet"]["tags"]) & set(keyword_arr))
        i+=1

    return 1/float(np.sum(key_calc_arr)+1)*(key_calc_arr)

def mediumKeywords(keywords):
    key_calc_arr=np.zeros(len(title_to_index))
    i=0
    if(keywords!=""):
        keyword_arr = keywords.split(",")
    else:
        keyword_arr = [keywords]
    for article in medium_data:
        if "tags" in article.keys():
            key_calc_arr[i]=len(set(article["tags"]) & set(keyword_arr))
        i+=1
    return 1/float(np.sum(key_calc_arr)+1)*(key_calc_arr)

def youtubeComments():
    comment_score_arr = np.zeros(len(yt_index_to_id))
    i=0
    for youtube in yt_data:
        yt_id = youtube['id']
        has_comments = (yt_id_to_comment[yt_id] != "")
        has_tags = (yt_id in yt_id_to_tags.keys())
        if (has_comments and has_tags):
            comments = set(tokenize(yt_id_to_comment[yt_id]))
            tags = set(youtube["snippet"]["tags"])
            comment_score_arr[i] = len(comments & tags)
        i+=1
    return comment_score_arr

def mediumComments():
    comment_score_arr = np.zeros(len(title_to_index))
    i=0
    for article in medium_data:
        title = article["title"]
        has_comments = (len(article["comments"]) > 0)
        has_tags = (title in title_to_tags.keys())
        if (has_comments and has_tags):
            comments = set()
            for comment in article["comments"]:
                comments.update(tokenize(comment))
            tags = set(title_to_tags[title])
            comment_score_arr[i] = len(comments & tags)
        i+=1
    return comment_score_arr

med_comment_weight = 0.1
yt_comment_weight = 0.01
keyword_weight = 0.1

#YouTube video comment scores
yt_comment_scores = yt_comment_weight*youtubeComments()
#Medium article comment scores
med_comment_scores = med_comment_weight*mediumComments()

#search function from YouTube video to Medium article
def mediumSearch(query,keywords):
    num_results = 10
    k_val = 100
    vid_id = url_to_id(query)
    api_response = get_single_video(vid_id)
    my_video_info = api_response['items'][0]
    my_title = my_video_info['snippet']['title']
    tags=" "
    if 'tags' in youtube["snippet"].keys():
        for tag in youtube["snippet"]["tags"]:
            tags=tag+" "

    query_vec = tfidf_vec.transform([my_title + tags]).toarray()

    return_arr = []
    
    svd_docs = SVD(medium_articles_by_vocab, query_vec, k_val)
    weighted_keywords = keyword_weight*mediumKeywords(keywords)
    sims = np.array(cosine_sim(svd_docs[1],svd_docs[0])).flatten()+weighted_keywords+med_comment_scores
    sort_idx = np.flip(np.argsort(sims))
    
    for i in range(0,num_results):
        article = medium_data[sort_idx[i]]
        return_arr.append((article["title"]+" "+str(sims[sort_idx[i]]), article["link"], article["comments"][0] if len(article["comments"])>0 else "",int(claps_to_nums(article["claps"])),article["reading_time"]))

    clap_arr = []
    for j in range(0,num_results):
        art_index = sort_idx[j]
        claps=medium_data[art_index]["claps"]
        claps_to_nums(claps)
        clap_arr.append(claps_to_nums(claps))

    clap_return_arr=[]
    for k in range(0,num_results):
        clap_return_arr.append(return_arr[np.argmax(clap_arr)])
        clap_arr[np.argmax(clap_arr)]=0

    return clap_return_arr

#search function from Medium article to YouTube video
def youtubeSearch(query,keywords):
    num_results = 10
    k_val = 200
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
    svd_docs = SVD(yt_vids_by_vocab, query_vec, k_val)
    weighted_keywords = keyword_weight*youtubeKeywords(keywords)
    sims = np.array(cosine_sim(svd_docs[1],svd_docs[0])).flatten()+weighted_keywords+yt_comment_scores
    sort_idx = np.flip(np.argsort(sims))
    id_arr = []

    for i in range(0,num_results):
        curr_id = yt_index_to_id[sort_idx[i]]
        return_arr.append((yt_id_to_title[curr_id]+" "+str(sims[sort_idx[i]]),"https://www.youtube.com/watch?v="+curr_id, yt_id_to_comment[curr_id][0][0] if len(yt_id_to_comment[curr_id])>0 and len(yt_id_to_comment[curr_id][0])>0 else "", yt_id_to_likes[curr_id], round(yt_id_to_length[curr_id])))
        id_arr.append(curr_id)


    return return_arr
    
def getLink(query):
    if(query == ""):
        return 0
    else:
        return mediumSearch(query)
