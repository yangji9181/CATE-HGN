import os
import json
import pickle
import datetime
import numpy as np
from collections import defaultdict
from statistics import mean
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
    

def process():
    print('%s-initializing data preprocess.' % datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))

    paper_sig = {}
    author_count = {}
    author_sig = {}
    author_cite = {}
    venue_count = {}
    venue_sig = {}
    venue_cite = {}
    term_count = {}
    term_sig = {}
    term_cite = {}

    keywords = ['data', 'learn', 'system', 'bio', 'vision', 'language', 'robotic', 'network', 'secur']
    print_keywords = ['data', 'learn', 'system', 'bio']

    for w in keywords:
        paper_sig[w] = defaultdict(int)
        author_count[w] = defaultdict(int)
        author_sig[w] = defaultdict(int)
        author_cite[w] = defaultdict(int)
        venue_count[w] = defaultdict(int)
        venue_sig[w] = defaultdict(int)
        venue_cite[w] = defaultdict(int)
        term_count[w] = defaultdict(int)
        term_sig[w] = defaultdict(int)
        term_cite[w] = defaultdict(int)

    print('%s-processing dblp raw data.' % datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))
    n_lines = 0
    with open('dblp/dblp.json', 'r') as f:
        line = f.readline()
        while line:
            n_lines += 1
            if n_lines % 100000 == 0:
                print('processing %d/3000000 lines' % n_lines)
                #if n_lines == 100000:
                #    break
            j = json.loads(line)
            if '_id' in j and 'venue' in j and 'n_citation' in j and 'year' in j and 'references' in j and 'authors' in j and 'title' in j and 'keywords' in j:
                n_cite_per_year = int(j['n_citation']['$numberInt'])/(2019-int(j['year']['$numberInt']))
                terms = word_tokenize(j['title'])
                if 'keywords' in j:
                    for phrase in j['keywords']:
                        terms += word_tokenize(phrase)
                if 'mag_fos' in j:
                    for phrase in j['mag_fos']:
                        terms += word_tokenize(phrase)
                if 'phrases' in j:
                    phrases = [phrase[8:-9].replace('_', ' ') for phrase in j['phrases']]
                    for phrase in phrases:
                        terms += word_tokenize(phrase)
                ps = PorterStemmer()
                terms = [ps.stem(term) for term in terms if term.isalpha() and term not in stopwords.words('english')]

                for w in keywords:
                    flag = False
                    for term in terms:
                        if term.count(ps.stem(w)) > 0:
                            flag = True
                            break
                    if flag:
                        if int(j['year']['$numberInt']) > 2015:
                            paper_sig[w][j['title']] += n_cite_per_year
                        for author in j['authors']:
                            if n_cite_per_year >= 5:
                                author_cite[w][author] += 3
                            elif n_cite_per_year <= 1:
                                author_cite[w][author] -= 3
                            author_sig[w][author] += np.log(n_cite_per_year+1)
                            author_count[w][author] += 1
                        if n_cite_per_year >= 5:
                            venue_cite[w][j['venue']] += 3
                        elif n_cite_per_year <= 1:
                            venue_cite[w][j['venue']] -= 3
                        venue_sig[w][j['venue']] += np.log(n_cite_per_year+1)
                        venue_count[w][j['venue']] += 1
                        for term in terms:
                            if n_cite_per_year >= 5:
                                term_cite[w][term] += 3
                            elif n_cite_per_year <= 1:
                                term_cite[w][term] -= 3
                            term_sig[w][term] += np.log(n_cite_per_year+1)
                            term_count[w][term] += 1
                        for w2 in keywords:
                            if w2 != w and n_cite_per_year >= 5:
                                for author in j['authors']:
                                    author_cite[w2][author] -= 1
                                venue_cite[w2][j['venue']] -= 1
                                for term in terms:
                                    term_cite[w2][term] -= 1
                
            line = f.readline()

    for w in print_keywords:
        sorted_words = sorted(paper_sig[w].items(), key=lambda x: x[1], reverse=True)
        for i in range(min(len(sorted_words), 30)):
            if sorted_words[i][1] > 0:
                print('%s-%s-%s: %d' % (w, 'sig paper', sorted_words[i][0], sorted_words[i][1]))

        sorted_words = sorted(author_cite[w].items(), key=lambda x: x[1], reverse=True)
        for i in range(min(len(sorted_words), 30)):
            if sorted_words[i][1] > 0:
                print('%s-%s-%s: %d' % (w, 'cite author', sorted_words[i][0], round(np.log(sorted_words[i][1])*10)))
        sorted_words = sorted(author_sig[w].items(), key=lambda x: x[1], reverse=True)
        for i in range(min(len(sorted_words), 30)):
            if sorted_words[i][1] > 0:
                print('%s-%s-%s: %d' % (w, 'sig author', sorted_words[i][0], round(np.log(sorted_words[i][1])*10)))
        sorted_words = sorted(author_count[w].items(), key=lambda x: x[1], reverse=True)
        for i in range(min(len(sorted_words), 30)):
            if sorted_words[i][1] > 0:
                print('%s-%s-%s: %d' % (w, 'count author', sorted_words[i][0], round(np.log(sorted_words[i][1])*10)))

        sorted_words = sorted(venue_cite[w].items(), key=lambda x: x[1], reverse=True)
        for i in range(min(len(sorted_words), 30)):
            if sorted_words[i][1] > 0:
                print('%s-%s-%s: %d' % (w, 'cite venue', sorted_words[i][0], round(np.log(sorted_words[i][1])*10)))
        sorted_words = sorted(venue_sig[w].items(), key=lambda x: x[1], reverse=True)
        for i in range(min(len(sorted_words), 30)):
            if sorted_words[i][1] > 0:
                print('%s-%s-%s: %d' % (w, 'sig venue', sorted_words[i][0], round(np.log(sorted_words[i][1])*10)))
        sorted_words = sorted(venue_count[w].items(), key=lambda x: x[1], reverse=True)
        for i in range(min(len(sorted_words), 30)):
            if sorted_words[i][1] > 0:
                print('%s-%s-%s: %d' % (w, 'count venue', sorted_words[i][0], round(np.log(sorted_words[i][1])*10)))

        print('%s-%s----------------------------------------------------------------' % (w, 'cite term', ))
        sorted_words = sorted(term_cite[w].items(), key=lambda x: x[1], reverse=True)
        for i in range(min(len(sorted_words), 100)):
            if sorted_words[i][1] > 0:
                print('%d,%s' % (round(np.log(sorted_words[i][1])*10), sorted_words[i][0]))
        print('%s-%s----------------------------------------------------------------' % (w, 'sig term', ))
        sorted_words = sorted(term_sig[w].items(), key=lambda x: x[1], reverse=True)
        for i in range(min(len(sorted_words), 100)):
            if sorted_words[i][1] > 0:
                print('%d,%s' % (round(np.log(sorted_words[i][1])*10), sorted_words[i][0]))
        print('%s-%s----------------------------------------------------------------' % (w, 'count term', ))
        sorted_words = sorted(term_count[w].items(), key=lambda x: x[1], reverse=True)
        for i in range(min(len(sorted_words), 100)):
            if sorted_words[i][1] > 0:
                print('%d,%s' % (round(np.log(sorted_words[i][1])*10), sorted_words[i][0]))

    with open('dblp/term.pkl', 'wb') as f:
        pickle.dump(term_cite, f)

def count():
    print('%s-initializing data preprocess.' % datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))

    high = 0
    low = 0
    count = 0

    print('%s-processing dblp raw data.' % datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))
    n_lines = 0
    with open('dblp/dblp.json', 'r') as f:
        line = f.readline()
        while line:
            n_lines += 1
            if n_lines % 100000 == 0:
                print('processing %d/3000000 lines' % n_lines)
                #if n_lines == 100000:
                #    break
            j = json.loads(line)
            if '_id' in j and 'venue' in j and 'n_citation' in j and 'year' in j and 'references' in j and 'authors' in j and 'title' in j and 'keywords' in j:
                n_cite_per_year = int(j['n_citation']['$numberInt'])/(2019-int(j['year']['$numberInt']))
                if n_cite_per_year <= 1:
                    low += 1
                elif n_cite_per_year >=4:
                    high += 1
                count += 1
            line = f.readline()

    print(count, high, low)

def tc():
    num_t = 100
    with open('dblp/term.pkl', 'rb') as f:
        term_sig = pickle.load(f)

    keywords = ['data', 'learn', 'system', 'bio']
    ps = PorterStemmer()
    dis_sum = 0
    for w in keywords:
        dis = 0
        count = 0
        sorted_words = sorted(term_sig[w].items(), key=lambda x: x[1], reverse=True)
        terms = [i[0] for i in sorted_words[0:num_t]]
        embs = np.zeros(shape=(num_t, 300), dtype=np.dtype(float))
        with open('dblp/glove.42B.300d.txt', 'r') as f:
            for line in f:
                tokens = line.strip().split()
                word = tokens[0]
                if ps.stem(word) in terms:
                    embs[terms.index(ps.stem(word))] += np.array([float(value) for value in tokens[1:]])
        for t1 in range(num_t):
            for t2 in range(num_t):
                if np.linalg.norm(embs[t1]) > 0 and np.linalg.norm(embs[t2]) > 0 and t1 != t2:
                    dis += np.linalg.norm(embs[t1] / np.linalg.norm(embs[t1]) - embs[t2] / np.linalg.norm(embs[t2]))
                    count += 1
        dis_avg = dis / count
        print("domain: %s: %f" % (w, dis_avg))
        dis_sum += dis_avg

    print("avg tc: %f" % (dis_sum/4))


def randtc():
    with open('dblp/term.pkl', 'rb') as f:
        term_sig = pickle.load(f)

    keywords = ['data', 'learn', 'system', 'bio']
    ps = PorterStemmer()
    dis_sum = 0
    terms = []
    for w in keywords:
        terms += list(term_sig[w].keys())[:300]
    terms = list(set(terms))
    embs = np.zeros(shape=(len(terms), 300), dtype=np.dtype(float))
    dis = 0
    count = 0
    with open('dblp/glove.42B.300d.txt', 'r') as f:
        for line in f:
            tokens = line.strip().split()
            word = tokens[0]
            if ps.stem(word) in terms:
                embs[terms.index(ps.stem(word))] += np.array([float(value) for value in tokens[1:]])
    for t1 in range(len(terms)):
        for t2 in range(len(terms)):
            if np.linalg.norm(embs[t1]) > 0 and np.linalg.norm(embs[t2]) > 0 and t1 != t2:
                dis += np.linalg.norm(embs[t1] / np.linalg.norm(embs[t1]) - embs[t2] / np.linalg.norm(embs[t2]))
                count += 1
    print(dis/count)

if __name__ == '__main__':
    process()



