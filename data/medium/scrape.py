# based on https://github.com/Hsankesara/medium-scrapper/blob/master/scrap.py
import urllib3
from bs4 import BeautifulSoup
import requests
import os
import csv
import json
import unicodedata
import pandas as pd
import time
import polyglot
from polyglot.text import Text, Word
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.common.exceptions import TimeoutException
from selenium.common.exceptions import NoSuchElementException
from selenium.common.exceptions import ElementNotInteractableException
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

chrome_options = Options()
chrome_options.add_argument('--auto-open-devtools-for-tabs')
# chrome_options.add_argument('--headless')

possible_tags = []

with open('medium_top_1000_tags.csv', 'r', encoding="utf8") as csvfile:
    reader = csv.DictReader(csvfile, delimiter=',')
    for row in reader:
        possible_tags.append(row['tag_name'].lower())

possible_tags = possible_tags[750:]
possible_tags = set(possible_tags)

def get_link(url, links):
    data = requests.get(url)
    soup = BeautifulSoup(data.content, 'html.parser')
    articles = soup.findAll('div', {"class": "postArticle-readMore"})
    for i in articles:
        link = i.a.get('href')
        if '?' in link:
            link = link[:link.rfind('?source=')]
        links.add(link)
    return soup


def get_links(tag, suffix):
    url = 'https://medium.com/tag/' + tag
    urls = [url + '/' + s for s in suffix]
    links = set()
    for url in urls:
        soup = get_link(url, links)
    return links


def get_links_and_related(tag, suffix, related_tags):
    url = 'https://medium.com/tag/' + tag
    urls = [url + '/' + s for s in suffix]
    links = set()
    for url in urls:
        soup = get_link(url, links)
        related = soup.findAll('a', {"data-action-source": "related"})
        for t in related:
            related_tags.add(t.get_text())
    return links

def get_articles(articles, links):
    driver = webdriver.Chrome(chrome_options=chrome_options)
    for link in links:
        try:
            article = {}
            while True:
                driver.get(link)
                wait = WebDriverWait(driver, 2)
                try:
                    dismissButton = wait.until(EC.element_to_be_clickable((By.XPATH, '//div[@class="postMeterBar u-height140 u-width100pct u-fixed u-overflowHidden u-backgroundWhite u-boxShadowTop u-borderBottomLightest u-bottom0 js-meterBanner"]/div/div[@class="u-xs-show u-absolute u-top0 u-right5"]/button[@data-action="dismiss-meter"]')))
                    dismissButton.send_keys(Keys.ENTER)
                except TimeoutException:
                    pass
                try:
                    driver.find_element(By.XPATH, '//button[@data-action="overlay-close"]').send_keys(Keys.ENTER)
                except NoSuchElementException:
                    pass
                try:
                    button = driver.find_element(By.XPATH, '//button[@data-action="show-other-responses"]')
                    button.send_keys(Keys.ENTER)
                    break
                except ElementNotInteractableException:
                    last_height = driver.execute_script("return document.body.scrollHeight")
                    while True:
                        # Scroll down to bottom
                        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")

                        # Wait to load page
                        time.sleep(0.75)

                        # Calculate new scroll height and compare with last scroll height
                        new_height = driver.execute_script("return document.body.scrollHeight")
                        if new_height == last_height:
                            break
                        last_height = new_height
                    try:
                        button = driver.find_element(By.XPATH, '//button[@data-action="show-other-responses"]')
                        button.send_keys(Keys.ENTER)
                        break
                    except ElementNotInteractableException:
                        break
                except NoSuchElementException:
                    break
            soup = BeautifulSoup(driver.page_source, 'lxml')
            title = soup.findAll('title')[0]
            title = title.get_text()
            art = soup.find('article')
            if art:
                if art.get('lang') != 'en':
                    continue
                else:
                    text = Text(title)
                    if not text.language.code == 'en':
                        continue
            author = soup.findAll('meta', {"name": "author"})[0]
            author = author.get('content')
            tags = soup.findAll('ul', 'tags')
            if tags:
                tags = {t.findAll('a')[0].get_text().lower() for t in tags[0]}
                article['tags'] = list(tags)
            else:
                article['tags'] = []
            article['author'] = unicodedata.normalize('NFKD', author)
            claps = soup.find('button', {"data-action": "show-recommends"})
            if claps:
                article['claps'] = unicodedata.normalize('NFKD', claps.get_text())
            else:
                article['claps'] = 0
            article['link'] = link
            article['title'] = unicodedata.normalize('NFKD', title)
            try:
                reading_time = int(soup.findAll('span', {"class": "readingTime"})[0].get('title').split()[0])
                article['reading_time'] = reading_time
            except:
                article['reading_time'] = 0
            body = soup.find('div', {"class": "postArticle-content"})
            paras = body.findAll('p')
            text = ''
            nxt_line = '\n'
            for para in paras:
                text += unicodedata.normalize('NFKD',
                                              para.get_text()) + nxt_line
            article['text'] = text
            responses = soup.findAll('div', {"class": "responsesStream"})
            article['comments'] = ''
            for r in responses:
                article['comments'] += r.get_text()
            num_comments = soup.find('button', {"class": "button button--chromeless u-baseColor--buttonNormal u-marginRight12", "data-action": "scroll-to-responses"})
            if num_comments and num_comments.get_text():
                article['num_comments'] = int(num_comments.get_text().replace(',', ''))
            else:
                article['num_comments'] = 0
            # for response in responses:
            #     response = response.findAll('div', {"class": "streamItem"})
            #     response = [r.find('div', {"class": "section-content"}) for r in response]
            #     response = [r.get_text() for r in response if r]
            #     article['comments'] += response
            articles.append(article)
        except KeyboardInterrupt:
            print('Exiting')
            os._exit(status=0)
        except:
            print(link)
    driver.quit()


def save_articles_json(articles, filename, is_write=True):
    if not is_write:
        try:
            with open(filename, 'r') as json_file:
                articles = json.load(json_file) + articles
        except FileNotFoundError:
            pass
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
    is_write = False
    # tags = input('Write tags in space separated format.\n')
    # tags = tags.split(' ')
    # tags = possible_tags
    # print(len(tags))
    file_name = input('Write destination file name.\n')
    if len(file_name.split('.')) == 1:
        file_name += '.json'
    suffixes = ['']
    articles = []
    related_tags = set()
    for i, tag in enumerate(possible_tags):
        if i % 50 == 0:
            save_articles_json(articles, file_name, is_write)
            articles = []
        links = get_links_and_related(tag, suffixes, related_tags)
        get_articles(articles, links)
        print('{}: {}'.format(i, tag))
    # related_tags = list(related_tags)
    # for i, tag in enumerate(related_tags):
    #     if i % 100 == 0:
    #         save_articles_json(articles, file_name, is_write)
    #         articles = []
    #     links = get_links(tag, suffixes)
    #     get_articles(articles, links)
    #     print('{}: {}'.format(i, tag))
    # To remove duplicates
    save_articles_json(articles, file_name, is_write)
    articles = pd.read_json(file_name, file_name)
    articles.drop_duplicates(subset=['author', 'claps', 'reading_time', 'text', 'title', 'link'], inplace=True)
    articles.to_json('deduped_' + file_name, orient='records')


if __name__ == '__main__':
    main()
