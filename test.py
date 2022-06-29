from collections import defaultdict
import numpy as np
import pickle


with open('dblp/data_unlabeled.pkl', 'rb') as f:
    data = pickle.load(f)


print('''
            n_lines: %d
            n_papers: %d
            n_authors: %d
            n_venues: %d
            n_terms: %d
            n_paper_paper: %d
            n_paper_author: %d
            n_paper_venue: %d
            n_paper_term: %d'''
            % (data['n_lines'], data['n_papers'], data['n_authors'], data['n_venues'], data['n_terms'], data['n_paper_paper'], data['n_paper_author'], data['n_paper_venue'], data['n_paper_term']))


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


#labels = labels // 10
unique, counts = np.unique(data['paper_label'], return_counts=True)

print(data['paper_label'].shape)
print(counts[0:2].sum())
print(counts[5:].sum())

#import matplotlib
#import matplotlib.pyplot as plt
#plt.bar(unique, np.log(counts+1))
#plt.show()


labels = []
index = []

low = 1
high = 5
for i in range(data['paper_label'].shape[0]):
    if data['paper_label'][i] <= low:
        labels.append(0)
        index.append(i)
    elif data['paper_label'][i] >= high:
        labels.append(1)
        index.append(i)
    else:
        labels.append(2)

print(len(index), len(labels))
unique, counts = np.unique(labels, return_counts=True)
print(unique, counts)


data['paper_label'] = np.array(labels, dtype=np.dtype(int))

rand_indices = np.random.permutation(np.array(index, dtype=np.dtype(int)))
n_test = len(index) // 5
n_val = len(index) // 100
data['test'] = rand_indices[0:n_test]
data['val'] = rand_indices[n_test:(n_test+n_val)]
data['train'] = rand_indices[(n_test+n_val):]

print(data['train'].shape, data['val'].shape, data['test'].shape)
with open('dblp/data.pkl', 'wb') as f:
    pickle.dump(data, f)


