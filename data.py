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
    

def glove(words, glove_emb):
    feat = np.zeros(shape=(1, 300), dtype=np.dtype(float))
    n = 0
    for word in words:
        if word in glove_emb:
            n += 1
            feat += glove_emb[word]
    if n > 0:
        feat = feat / n
    return feat

def preprocess():
    print('%s-initializing data preprocess.' % datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))

    data = {}

    data['n_feat'] = 300
    data['n_lines'] = 0
    data['n_papers'] = 0
    data['n_authors'] = 0
    data['n_venues'] = 0
    data['n_terms'] = 0
    data['n_with_keywords'] = 0
    data['n_num_keywords'] = 0
    data['n_paper_paper'] = 0
    data['n_paper_author'] = 0
    data['n_paper_venue'] = 0
    data['n_paper_term'] = 0

    data['papers'] = {}

    data['paper_map'] = {}
    data['author_map'] = {}
    data['venue_map'] = {}
    data['term_map'] = {}

    data['paper_paper_adj'] = defaultdict(set)
    data['paper_author_adj'] = defaultdict(set)
    data['paper_venue_adj'] = defaultdict(set)
    data['paper_term_adj'] = defaultdict(set)
    data['author_paper_adj'] = defaultdict(set)
    data['venue_paper_adj'] = defaultdict(set)
    data['term_paper_adj'] = defaultdict(set)

    paper_words = defaultdict(set)
    venue_words = defaultdict(set)
    paper_paper_adj_ = defaultdict(set)
    author_author_adj = defaultdict(set)
    paper_label_ = []
    vocab = set()

    venues = set()

    venue_domain = {}
    with open('dblp/venues_selected.txt', 'r') as f:
        line = f.readline()
        while line:
            if len(line.split(':')) == 2:
                venue, domain = line.split(':')
                venue_domain[venue] = domain.strip()
            line = f.readline()

    print('%s-processing dblp raw data.' % datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))
    with open('dblp/dblp.json', 'r') as f:
        line = f.readline()
        while line:
            data['n_lines'] += 1
            if data['n_lines'] % 100000 == 0:
                print('processing %d/3000000 lines' % data['n_lines'])
                # break
            j = json.loads(line)
            #if '_id' in j and 'venue' in j and j['venue'] in venue_domain and 'n_citation' in j and 'year' in j and 'references' in j and 'authors' in j and 'title' in j:
            if '_id' in j and 'venue' in j and 'n_citation' in j and 'year' in j and 'references' in j and 'authors' in j and 'title' in j:
                if 'data' in j['venue'].strip().lower().split():
                    venues.add(j['venue'])
                else:
                    line = f.readline()
                    continue
                data['n_papers'] += 1
                paper_id = data['n_papers'] - 1
                data['papers'][j['_id']] = {}
                data['papers'][j['_id']]['n_cite'] = int(j['n_citation']['$numberInt'])
                data['papers'][j['_id']]['n_author'] = len(j['authors'])
                data['papers'][j['_id']]['year'] = int(j['year']['$numberInt'])
                data['papers'][j['_id']]['venue'] = j['venue'].lower()
                #data['papers'][j['_id']]['domain'] = venue_domain[j['venue']]
                data['papers'][j['_id']]['id'] = paper_id
                
                data['paper_map'][j['_id']] = paper_id
                n_cite_per_year = int(j['n_citation']['$numberInt'])/(2019-int(j['year']['$numberInt']))
                if int(j['year']['$numberInt']) > 2010:
                    paper_label_.append(n_cite_per_year)
                else:
                    paper_label_.append(-1)
                    
                for word in j['title']:
                    paper_words[paper_id].add(word.lower())
                    vocab.add(word.lower())

                for r in j['references']:
                    paper_paper_adj_[paper_id].add(r)

                data['papers'][j['_id']]['authors'] = []
                for a in j['authors']:
                    if a.lower() not in data['author_map']:
                        data['n_authors'] += 1
                        author_id = data['n_authors'] - 1
                        data['author_map'][a.lower()] = author_id
                    else:
                        author_id = data['author_map'][a.lower()]
                    data['paper_author_adj'][paper_id].add(author_id)
                    data['author_paper_adj'][author_id].add(paper_id)
                    data['n_paper_author'] += 1
                for a in j['authors']:
                    for b in j['authors']:
                        if a != b:
                            author_author_adj[data['author_map'][a.lower()]].add(data['author_map'][b.lower()])
                
                if j['venue'].lower() not in data['venue_map']:
                    data['n_venues'] += 1
                    venue_id = data['n_venues'] - 1
                    data['venue_map'][j['venue'].lower()] = venue_id
                else:
                    venue_id = data['venue_map'][j['venue'].lower()]
                data['paper_venue_adj'][paper_id].add(venue_id)
                data['venue_paper_adj'][venue_id].add(paper_id)
                data['n_paper_venue'] += 1
                for word in j['venue']:
                    venue_words[venue_id].add(word.lower())
                    vocab.add(word.lower())
                
                ps = PorterStemmer()
                #terms = word_tokenize(j['title'])
                terms = []
                #if 'abstract' in j:
                #   terms += word_tokenize(j['abstract'])
                if 'keywords' in j:
                    for phrase in j['keywords']:
                        terms += word_tokenize(phrase)
                    data['n_with_keywords'] += 1
                    data['n_num_keywords'] += len(j['keywords'])
                #if 'mag_fos' in j:
                #    for phrase in j['mag_fos']:
                #        terms += word_tokenize(phrase)
                #if 'phrases' in j:
                #    phrases = [phrase[8:-9].replace('_', ' ') for phrase in j['phrases']]
                #    for phrase in phrases:
                #        terms += word_tokenize(phrase)
                terms = [ps.stem(term) for term in terms if term.isalpha() and term not in stopwords.words('english')]
                for term in terms:
                    if term.lower() not in data['term_map']:
                        data['n_terms'] += 1
                        term_id = data['n_terms'] - 1
                        data['term_map'][term.lower()] = term_id
                    else:
                        term_id = data['term_map'][term.lower()]
                    data['paper_term_adj'][paper_id].add(term_id)
                    data['term_paper_adj'][term_id].add(paper_id)
                    data['n_paper_term'] += 1
                    vocab.add(term.lower())
                
            line = f.readline()

    data['paper_label'] = np.array(paper_label_, dtype=np.dtype(int))
    print('%s-reindex paper-paper links.' % datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))
    for paper_id_a in paper_paper_adj_:
        for paper_id_b_ in paper_paper_adj_[paper_id_a]:
            if paper_id_b_ in data['papers']:
                data['paper_paper_adj'][paper_id_a].add(data['papers'][paper_id_b_]['id'])
                data['n_paper_paper'] += 1

    '''
    print('%s-load glove embedding.' % datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))
    glove_emb = {}
    ps = PorterStemmer()
    with open('dblp/glove.42B.300d.txt', 'r') as f:
        for line in f:
            tokens = line.split()
            word = tokens[0]
            if ps.stem(word) in vocab:
                emb = np.array([float(value) for value in tokens[1:]])
                glove_emb[word] = emb

    print('%s-look up paper, venue, term embeddings.' % datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))
    data['paper_feat'] = np.zeros(shape=(data['n_papers'], 300), dtype=np.dtype(float))
    for paper_id in paper_words:
        data['paper_feat'][paper_id] = glove(paper_words[paper_id], glove_emb)
    data['venue_feat'] = np.zeros(shape=(data['n_venues'], 300), dtype=np.dtype(float))
    for venue_id in venue_words:
        data['venue_feat'][venue_id] = glove(venue_words[venue_id], glove_emb)
    data['term_feat'] = np.zeros(shape=(data['n_terms'], 300), dtype=np.dtype(float))
    for term in data['term_map']:
        data['term_feat'][data['term_map'][term]] = glove(term, glove_emb)

    print('%s-compute author embedding with deepwalk.' % datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))
    dw_emb_file = 'dblp/DW.EMB'
    dw_adj_file = 'dblp/DW.ADJ'
    with open(dw_adj_file, 'w') as f:
        for a in author_author_adj:
            for b in author_author_adj[a]:
                f.write('%d %d\n' % (a, b))
    os.system('deepwalk --input "%s" --output "%s" --number-walks 10 --representation-size 300 --walk-length 10 --window-size 5' % (dw_adj_file, dw_emb_file))

    data['author_feat'] = np.zeros(shape=(data['n_authors'], 300), dtype=np.dtype(float))
    with open(dw_emb_file, 'r') as f:
        line = f.readline()
        while line:
            tokens = line.strip().split(' ')
            if len(tokens) > 2:
                data['author_feat'][int(tokens[0])] = np.array(tokens[1:], dtype=float)
            line = f.readline()

    print('%s-save results.' % datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))
    with open('dblp/data.pkl', 'wb') as f:
        pickle.dump(data, f)
    '''


    print('''
            n_lines: %d
            n_papers: %d
            n_authors: %d
            n_venues: %d
            n_terms: %d
            n_with_keywords: %d
            n_num_keywords: %d
            n_paper_paper: %d
            n_paper_author: %d
            n_paper_venue: %d
            n_paper_term: %d'''
            % (data['n_lines'], data['n_papers'], data['n_authors'], data['n_venues'], data['n_terms'], data['n_with_keywords'], data['n_num_keywords'], data['n_paper_paper'], data['n_paper_author'], data['n_paper_venue'], data['n_paper_term']))


    #unique, counts = np.unique(data['paper_label'], return_counts=True)
    with open('dblp/venue.pkl', 'wb') as f:
        pickle.dump(venues, f)
        #pickle.dump(counts, f)

'''
    print(data['papers'])

    print(data['paper_map'])
    print(data['author_map'])
    print(data['venue_map'])
    print(data['term_map'])

    print(data['paper_feat'])
    print(data['author_feat'])
    print(data['venue_feat'])
    print(data['term_feat'])

    print(data['paper_paper_adj'])
    print(data['paper_author_adj'])
    print(data['paper_venue_adj'])
    print(data['paper_term_adj'])
    print(data['author_paper_adj'])
    print(data['venue_paper_adj'])
    print(data['term_paper_adj'])

    print(data['paper_label'])
'''


if __name__ == '__main__':
    preprocess()



