from collections import defaultdict
import time
from joblib import Parallel, delayed
from multiprocessing import cpu_count
from math import ceil
import torch
from torch import nn
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from utils import format_time, get_free_gpu, make_dataloader
from nltk.corpus import stopwords
from transformers import BertTokenizer, AdamW, get_linear_schedule_with_warmup
import numpy as np
import os
import shutil
from tqdm import tqdm
from model import LOTClass


class LOTClassTrainer(object):

    def __init__(self, args):
        self.args = args
        self.max_len = args.max_len
        # self.device = device
        self.dataset_dir = args.dataset_dir
        self.num_cpus = cpu_count() - 1 if cpu_count() > 1 else 1
        self.train_batch_size = args.batch_size
        self.eval_batch_size = 4 * args.batch_size
        self.pretrained_lm = 'bert-base-uncased'
        self.tokenizer = BertTokenizer.from_pretrained(self.pretrained_lm, do_lower_case=True)
        self.vocab = self.tokenizer.get_vocab()
        self.vocab_size = len(self.vocab)
        self.mask_id = self.vocab[self.tokenizer.mask_token]
        self.inv_vocab = {k:v for v, k in self.vocab.items()}
        self.read_label_names(args.dataset_dir, args.label_names_file)
        self.num_class = len(self.label_name_dict)
        self.model = LOTClass.from_pretrained(self.pretrained_lm, output_attentions=False, output_hidden_states=False, num_labels=self.num_class)
        self.read_data(args.dataset_dir, args.train_file, args.train_label_file, args.test_file, args.test_label_file)
        self.world_size = args.world_size
        self.temp_dir = 'tmp'
        self.mtp_loss = nn.CrossEntropyLoss()
        # self.device = torch.device("cuda")
        # self.model.to(torch.device("cuda"))
        # self.num_gpus = torch.cuda.device_count()
        # print(f"Using {self.num_gpus} GPU(s) in total.")
        # if self.num_gpus > 1:
        #     self.model = nn.DataParallel(self.model)

    # get document truncation statistics with the defined max length
    def corpus_trunc_stats(self, docs):
        doc_len = []
        for doc in docs:
            input_ids = self.tokenizer.encode(doc, add_special_tokens=True)
            doc_len.append(len(input_ids))
        print(f"Document max length: {np.max(doc_len)}, avg length: {np.mean(doc_len)}, std length: {np.std(doc_len)}")
        trunc_frac = np.sum(np.array(doc_len) > self.max_len) / len(doc_len)
        print(f"Truncated fraction of all documents: {trunc_frac}")

    # convert a list of strings to token ids
    def encode(self, docs):
        encoded_dict = self.tokenizer.batch_encode_plus(docs, add_special_tokens=True, max_length=self.max_len, padding='max_length',
                                                        return_attention_mask=True, truncation=True, return_tensors='pt')
        input_ids = encoded_dict['input_ids']
        attention_masks = encoded_dict['attention_mask']
        return input_ids, attention_masks

    # convert list of token ids to list of strings
    def decode(self, ids):
        strings = self.tokenizer.batch_decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        return strings

    # convert dataset into tensors
    def create_dataset(self, dataset_dir, text_file, label_file, loader_name, find_label_name=False, label_name_loader_name=None):
        loader_file = os.path.join(dataset_dir, loader_name)
        if os.path.exists(loader_file):
            print(f"Loading encoded texts from {loader_file}")
            data = torch.load(loader_file)
        else:
            print(f"Reading texts from {os.path.join(dataset_dir, text_file)}")
            corpus = open(os.path.join(dataset_dir, text_file), encoding="utf-8")
            docs = [doc.strip() for doc in corpus.readlines()]
            print(f"Reading labels from {os.path.join(dataset_dir, label_file)}")
            truth = open(os.path.join(dataset_dir, label_file))
            labels = [int(label.strip()) for label in truth.readlines()]
            labels = torch.tensor(labels)
            print(f"Converting texts into tensors.")
            chunk_size = ceil(len(docs) / self.num_cpus)
            chunks = [docs[x:x+chunk_size] for x in range(0, len(docs), chunk_size)]
            results = Parallel(n_jobs=self.num_cpus)(delayed(self.encode)(docs=chunk) for chunk in chunks)
            input_ids = torch.cat([result[0] for result in results])
            attention_masks = torch.cat([result[1] for result in results])
            print(f"Saving encoded texts into {loader_file}")
            data = {"input_ids": input_ids, "attention_masks": attention_masks, "labels": labels}
            torch.save(data, loader_file)
        if find_label_name:
            loader_file = os.path.join(dataset_dir, label_name_loader_name)
            if os.path.exists(loader_file):
                print(f"Loading texts with label names from {loader_file}")
                label_name_data = torch.load(loader_file)
            else:
                print(f"Reading texts from {os.path.join(dataset_dir, text_file)}")
                corpus = open(os.path.join(dataset_dir, text_file), encoding="utf-8")
                docs = [doc.strip() for doc in corpus.readlines()]
                print("Locating label names in the corpus.")
                chunk_size = ceil(len(docs) / self.num_cpus)
                chunks = [docs[x:x+chunk_size] for x in range(0, len(docs), chunk_size)]
                results = Parallel(n_jobs=self.num_cpus)(delayed(self.label_name_occurrence)(docs=chunk) for chunk in chunks)
                input_ids_with_label_name = torch.cat([result[0] for result in results])
                attention_masks_with_label_name = torch.cat([result[1] for result in results])
                label_name_idx = torch.cat([result[2] for result in results])
                assert len(input_ids_with_label_name) > 0, "No label names appear in corpus!"
                label_name_data = {"input_ids": input_ids_with_label_name, "attention_masks": attention_masks_with_label_name, "labels": label_name_idx}
                loader_file = os.path.join(dataset_dir, label_name_loader_name)
                print(f"Saving texts with label names into {loader_file}")
                torch.save(label_name_data, loader_file)
            return data, label_name_data
        else:
            return data
    
    # find label name indices and replace out-of-vocab label names with [MASK]
    def label_name_in_doc(self, doc):
        doc = self.tokenizer.tokenize(doc)
        label_idx = -1 * torch.ones(self.max_len, dtype=torch.long)
        new_doc = []
        wordpcs = []
        idx = 1 # index starts at 1 due to [CLS] token
        for i, wordpc in enumerate(doc):
            wordpcs.append(wordpc[2:] if wordpc.startswith("##") else wordpc)
            if idx >= self.max_len - 1: # last index will be [SEP] token
                break
            if i == len(doc) - 1 or not doc[i+1].startswith("##"):
                word = ''.join(wordpcs)
                if word in self.label2class:
                    label_idx[idx] = self.label2class[word]
                    # replace label names that are not in tokenizer's vocabulary with the [MASK] token
                    if word not in self.vocab:
                        wordpcs = [self.tokenizer.mask_token]
                new_word = ''.join(wordpcs)
                if new_word != self.tokenizer.unk_token:
                    idx += len(wordpcs)
                    new_doc.append(new_word)
                wordpcs = []
        if (label_idx >= 0).any():
            return ' '.join(new_doc), label_idx
        else:
            return None

    # find label name occurrences in the corpus
    def label_name_occurrence(self, docs):
        text_with_label = []
        label_name_idx = []
        for doc in docs:
            result = self.label_name_in_doc(doc)
            if result is not None:
                text_with_label.append(result[0])
                label_name_idx.append(result[1].unsqueeze(0))
        if len(text_with_label) > 0:
            encoded_dict = self.tokenizer.batch_encode_plus(text_with_label, add_special_tokens=True, max_length=self.max_len, 
                                                            padding='max_length', return_attention_mask=True, truncation=True, return_tensors='pt')
            input_ids_with_label_name = encoded_dict['input_ids']
            attention_masks_with_label_name = encoded_dict['attention_mask']
            label_name_idx = torch.cat(label_name_idx, dim=0)
        else:
            input_ids_with_label_name = torch.ones(0, self.max_len, dtype=torch.long)
            attention_masks_with_label_name = torch.ones(0, self.max_len, dtype=torch.long)
            label_name_idx = torch.ones(0, self.max_len, dtype=torch.long)
        return input_ids_with_label_name, attention_masks_with_label_name, label_name_idx

    # read text corpus and labels from files
    def read_data(self, dataset_dir, train_file, train_label_file, test_file, test_label_file):
        self.train_data, self.label_name_data = self.create_dataset(dataset_dir, train_file, train_label_file, "train.pt", 
                                                                    find_label_name=True, label_name_loader_name="label_name_data.pt")
        if test_file is not None:
            self.test_data = self.create_dataset(dataset_dir, test_file, test_label_file, "test.pt")

    # read label names from file
    def read_label_names(self, dataset_dir, label_name_file):
        label_name_file = open(os.path.join(dataset_dir, label_name_file))
        label_names = label_name_file.readlines()
        self.label_name_dict = {i: [word.lower() for word in topic_words.strip().split()] for i, topic_words in enumerate(label_names)}
        print(f"Label names used for each class are: {self.label_name_dict}")
        self.label2class = {}
        self.all_label_name_ids = [self.mask_id]
        self.all_label_names = [self.tokenizer.mask_token]
        for class_idx in self.label_name_dict:
            for word in self.label_name_dict[class_idx]:
                assert word not in self.label2class, f"\"{word}\" used as the label name by multiple classes!"
                self.label2class[word] = class_idx
                if word in self.vocab:
                    self.all_label_name_ids.append(self.vocab[word])
                    self.all_label_names.append(word)

    # filter out stop words and words in multiple topics
    def filter_keywords(self, top_num=100):
        all_words = defaultdict(list)
        sorted_dicts = {}
        for i, cat_dict in self.topic_words_freq.items():
            sorted_dict = {k:v for k, v in sorted(cat_dict.items(), key=lambda item: item[1], reverse=True)[:top_num]}
            sorted_dicts[i] = sorted_dict
            for word_id in sorted_dict:
                all_words[word_id].append(i)
        repeat_words = []
        for word_id in all_words:
            if len(all_words[word_id]) > 1:
                repeat_words.append(word_id)
        self.topic_vocab = {}
        for i, sorted_dict in sorted_dicts.items():
            self.topic_vocab[i] = np.array(list(sorted_dict.keys()))
        stopwords_vocab = stopwords.words('english')
        for i, word_list in self.topic_vocab.items():
            delete_idx = []
            for j, word_id in enumerate(word_list):
                word = self.inv_vocab[word_id]
    #             if word in orig_seeds[i]:
    #                 continue
                if not word.isalpha() or len(word) == 1 or word in stopwords_vocab or word in repeat_words:
                    delete_idx.append(j)
            self.topic_vocab[i] = np.delete(self.topic_vocab[i], delete_idx)

    # construct topic vocabulary (distributed function)
    def category_keywords_dist(self, rank, top_pred_num=50, topic_vocab_size=100, loader_name="topic_vocab.pt"):
        dist.init_process_group(
            backend='nccl',
            init_method='tcp://localhost:12345',
            world_size=self.world_size,
            rank=rank
        )
        # create local model
        model = self.model.to(rank)
        model = DDP(model, device_ids=[rank])
        model.eval()
        dataset = TensorDataset(self.label_name_data["input_ids"], self.label_name_data["attention_masks"], self.label_name_data["labels"])
        train_sampler = DistributedSampler(dataset, num_replicas=self.world_size, rank=rank)
        label_name_dataset_loader = DataLoader(dataset, sampler=train_sampler, batch_size=self.eval_batch_size, shuffle=False)
        topic_words_freq = {i: defaultdict(float) for i in range(self.num_class)}
        wrap_label_name_dataset_loader = tqdm(label_name_dataset_loader) if rank == 0 else label_name_dataset_loader
        for batch in wrap_label_name_dataset_loader:
            with torch.no_grad():
                input_ids = batch[0].to(rank)
                input_mask = batch[1].to(rank)
                label_pos = batch[2].to(rank)
                match_idx = label_pos >= 0
                predictions = model(input_ids,
                                    pred_mode="mlm",
                                    token_type_ids=None, 
                                    attention_mask=input_mask)
                _, sorted_res = torch.topk(predictions[match_idx], top_pred_num, dim=-1)
                label_idx = label_pos[match_idx]
                for i, word_list in enumerate(sorted_res):
                    for j, word_id in enumerate(word_list):
                        topic_words_freq[label_idx[i].item()][word_id.item()] += 1
        save_file = os.path.join(self.temp_dir, f"{rank}_"+loader_name)
        torch.save(topic_words_freq, save_file)

    # construct topic vocabulary
    def category_keywords(self, top_pred_num=50, topic_vocab_size=100, loader_name="topic_vocab.pt"):
        loader_file = os.path.join(self.dataset_dir, loader_name)
        if os.path.exists(loader_file):
            print(f"Loading topic vocabulary from {loader_file}")
            self.topic_vocab = torch.load(loader_file)
        else:
            print("Contructing topic vocabulary.")
            if not os.path.exists(self.temp_dir):
                os.makedirs(self.temp_dir)
            mp.spawn(self.category_keywords_dist, nprocs=self.world_size, args=(top_pred_num, topic_vocab_size, loader_name))
            gather_res = []
            for f in os.listdir(self.temp_dir):
                if f[-3:] == '.pt':
                    gather_res.append(torch.load(os.path.join(self.temp_dir, f)))
            assert len(gather_res) == self.world_size, "Number of saved files not equal to number of processes!"
            self.topic_words_freq = {i: defaultdict(float) for i in range(self.num_class)}
            for i in range(self.num_class):
                for topic_words_freq in gather_res:
                    for word_id, freq in topic_words_freq[i].items():
                        self.topic_words_freq[i][word_id] += freq
            self.filter_keywords(top_num=topic_vocab_size)
            torch.save(self.topic_vocab, loader_file)
            if os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)
        for i, topic_vocab in self.topic_vocab.items():
            print(f"Class {i} topic vocabulary: {[self.inv_vocab[w] for w in topic_vocab]}")
            