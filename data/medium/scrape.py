# based on https://github.com/Hsankesara/medium-scrapper/blob/master/scrap.py
import urllib3
from bs4 import BeautifulSoup
import requests
import os
import csv
import json
import unicodedata
import pandas as pd


def get_links(tag, suffix):
    url = 'https://medium.com/tag/' + tag
    urls = [url + '/' + s for s in suffix]
    links = []
    for url in urls:
        data = requests.get(url)
        soup = BeautifulSoup(data.content, 'html.parser')
        articles = soup.findAll('div', {"class": "postArticle-readMore"})
        for i in articles:
            links.append(i.a.get('href'))
    return links


def get_articles(articles, links):
    for link in links:
        try:
            article = {}
            data = requests.get(link)
            soup = BeautifulSoup(data.content, 'html.parser')
            title = soup.findAll('title')[0]
            title = title.get_text()
            author = soup.findAll('meta', {"property": "author"})[0]
            author = author.get('content')
            article['author'] = unicodedata.normalize('NFKD', author)
            claps = soup.findAll(
                'button', {"data-action": "show-recommends"})[0].get_text()
            article['claps'] = unicodedata.normalize('NFKD', claps)
            article['link'] = link
            article['title'] = unicodedata.normalize('NFKD', title)
            reading_time = int(soup.findAll('span', {"class": "readingTime"})[
                               0].get('title').split()[0])
            article['reading_time'] = reading_time
            print(title)
            paras = soup.findAll('p')
            text = ''
            nxt_line = '\n'
            for para in paras:
                text += unicodedata.normalize('NFKD',
                                              para.get_text()) + nxt_line
            article['text'] = text
            articles.append(article)
        except KeyboardInterrupt:
            print('Exiting')
            os._exit(status=0)
        except:
            # for exceptions caused due to change of format on that page
            continue


def save_articles_json(articles, filename, is_write=True):
    with open(filename, 'w') as json_file:
        json.dump(articles, json_file)


def save_articles(articles, csv_file,  is_write=True):
    csv_columns = ['author', 'claps', 'reading_time', 'link', 'title', 'text']
    print(csv_file)
    if is_write:
        with open(csv_file, 'w') as csvfile:
            writer = csv.DictWriter(
                csvfile, fieldnames=csv_columns, delimiter=',')
            writer.writeheader()
            for data in articles:
                writer.writerow(data)
            csvfile.close()
    else:
        with open(csv_file, 'a+') as csvfile:
            writer = csv.DictWriter(
                csvfile, fieldnames=csv_columns,  delimiter=',')
            for data in articles:
                writer.writerow(data)
            csvfile.close()


def main():
    is_write = True
    # tags = input('Write tags in space separated format.\n')
    # tags = tags.split(' ')
    tags = ['ai', 'android', 'apple', 'architecture', 'art', 'artificial-intelligence',
            'big-data', 'bitcoin', 'blacklivesmatter', 'blockchain', 'blog', 'blogging',
            'books', 'branding', 'business', 'college', 'computer-science', 'creativity',
            'cryptocurrency', 'culture', 'data', 'data-science', 'data-visualization',
            'deep-learning', 'design', 'dogs', 'donald-trump', 'economics', 'education',
            'energy', 'entrepreneurship', 'environment', 'ethereum', 'feminism', 'fiction',
            'food', 'football', 'google', 'government', 'happiness', 'health', 'history',
            'humor', 'inspiration', 'investing', 'ios', 'javascript', 'jobs', 'journalism',
            'leadership', 'life', 'life-lessons', 'love', 'machine-learning', 'marketing',
            'medium', 'mobile', 'motivation', 'movies', 'music', 'nba', 'news', 'nutrition',
            'parenting', 'personal-development', 'photography', 'poem', 'poetry', 'politics',
            'product-design', 'productivity', 'programming', 'psychology', 'python', 'racism',
            'react', 'relationships', 'science', 'self-improvement', 'social-media',
            'software-engineering', 'sports', 'startup', 'tech', 'technology', 'travel',
            'trump', 'ux', 'venture-capital', 'web-design', 'web-development', 'women',
            'wordpress', 'work', 'writing']
    print(len(tags))
    file_name = input('Write destination file name.\n')
    if len(file_name.split('.')) == 1:
        file_name += '.json'
    suffixes = ['', 'latest']
    articles = []
    for tag in tags:
        links = get_links(tag, suffixes)
        get_articles(articles, links)
        is_write = False
    # To remove duplicates
    save_articles_json(articles, file_name, is_write)
    articles = pd.read_json(file_name, file_name)
    articles = articles.drop_duplicates()
    articles.to_json(file_name, orient='records')


if __name__ == '__main__':
    main()
