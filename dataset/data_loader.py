# -*- coding: utf-8 -*-

"""
ReferIt, UNC, UNC+ and GRef referring image segmentation PyTorch dataset.

Define and group batches of images, segmentations and queries.
Based on:
https://github.com/chenxi116/TF-phrasecut-public/blob/master/build_batches.py
"""

import os
import sys
import cv2
import json
import uuid
import tqdm
import math
import torch
import random
# import h5py
import numpy as np
import os.path as osp
import scipy.io as sio
import torch.utils.data as data
from collections import OrderedDict
sys.path.append('.')
import operator
import utils
from utils import Corpus

import argparse
import collections
import logging
import json
import re

from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.modeling import BertModel
from utils.transforms import letterbox, random_affine, random_affine2

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from strsimpy.normalized_levenshtein import NormalizedLevenshtein

sys.modules['utils'] = utils

cv2.setNumThreads(0)

def read_examples(input_line, unique_id):
    """Read a list of `InputExample`s from an input file."""
    examples = []
    line = input_line
    line = line.strip()
    text_a = None
    text_b = None
    m = re.match(r"^(.*) \|\|\| (.*)$", line)
    if m is None:
        text_a = line
    else:
        text_a = m.group(1)
        text_b = m.group(2)
    examples.append(
        InputExample(unique_id=unique_id, text_a=text_a, text_b=text_b))
    return examples

## Bert text encoding
class InputExample(object):
    def __init__(self, unique_id, text_a, text_b):
        self.unique_id = unique_id
        self.text_a = text_a
        self.text_b = text_b

class InputFeatures(object):
    """A single set of features of data."""
    def __init__(self, unique_id, tokens, input_ids, input_mask, input_type_ids):
        self.unique_id = unique_id
        self.tokens = tokens
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.input_type_ids = input_type_ids

def convert_examples_to_features(examples, seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""
    features = []
    for (ex_index, example) in enumerate(examples):
        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)

        if tokens_b:
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > seq_length - 2:
                tokens_a = tokens_a[0:(seq_length - 2)]
        tokens = []
        input_type_ids = []
        tokens.append("[CLS]")
        input_type_ids.append(0)
        for token in tokens_a:
            tokens.append(token)
            input_type_ids.append(0)
        tokens.append("[SEP]")
        input_type_ids.append(0)

        if tokens_b:
            for token in tokens_b:
                tokens.append(token)
                input_type_ids.append(1)
            tokens.append("[SEP]")
            input_type_ids.append(1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < seq_length:
            input_ids.append(0)
            input_mask.append(0)
            input_type_ids.append(0)

        assert len(input_ids) == seq_length
        assert len(input_mask) == seq_length
        assert len(input_type_ids) == seq_length
        features.append(
            InputFeatures(
                unique_id=example.unique_id,
                tokens=tokens,
                input_ids=input_ids,
                input_mask=input_mask,
                input_type_ids=input_type_ids))
    return features

class DatasetNotFoundError(Exception):
    pass

class ReferDataset(data.Dataset):
    SUPPORTED_DATASETS = {
        'referit': {'splits': ('train', 'val', 'trainval', 'test')},
        'unc': {
            'splits': ('train', 'val', 'trainval', 'testA', 'testB'),
            'params': {'dataset': 'refcoco', 'split_by': 'unc'}
        },
        'unc+': {
            'splits': ('train', 'val', 'trainval', 'testA', 'testB'),
            'params': {'dataset': 'refcoco+', 'split_by': 'unc'}
        },
        'gref': {
            'splits': ('train', 'val'),
            'params': {'dataset': 'refcocog', 'split_by': 'google'}
        },
        'gref_umd': {
            'splits': ('train', 'val', 'test'),
            'params': {'dataset': 'refcocog', 'split_by': 'umd'}
        },
        'flickr': {
            'splits': ('train', 'val', 'test')
        },
        'reftext': {
            'splits': ('train', 'val', 'test', 'subtest_street', 'subtest_shelf', 'subtest_home', 'subtest_sport', 'subtest_other', 'subtest_oov', 'subtest_semantic'),
            'params': {'dataset': 'reftext', 'split_by': ''}
        }
    }

    def __init__(self, data_root, split_root='data', dataset='referit', imsize=256,
                 transform=None, augment=False, return_idx=False, testmode=False,
                 split='train', max_query_len=128, lstm=False, bert_model='bert-base-uncased'):
        self.images = []
        self.data_root = data_root
        self.split_root = split_root
        self.dataset = dataset
        self.imsize = imsize
        self.query_len = max_query_len
        self.lstm = lstm
        self.transform = transform
        self.testmode = testmode
        self.split = split
        self.tokenizer = BertTokenizer.from_pretrained(bert_model, do_lower_case=True)
        self.augment=augment
        self.return_idx=return_idx

        if self.dataset == 'referit':
            self.dataset_root = osp.join(self.data_root, 'referit')
            self.im_dir = osp.join(self.dataset_root, 'images')
            self.split_dir = osp.join(self.dataset_root, 'splits')
        elif self.dataset == 'flickr':
            self.dataset_root = osp.join(self.data_root, 'Flickr30k')
            self.im_dir = osp.join(self.dataset_root, 'flickr30k_images')
        elif self.dataset == 'reftext':
            self.dataset_root = osp.join(self.data_root, 'other')
            self.im_dir = osp.join(self.dataset_root, 'images', 'reftext')
        else:   ## refcoco, etc.
            self.dataset_root = osp.join(self.data_root, 'other')
            self.im_dir = osp.join(
                self.dataset_root, 'images', 'mscoco', 'images', 'train2014')
            self.split_dir = osp.join(self.dataset_root, 'splits')

        if not self.exists_dataset():
            # self.process_dataset()
            print('Please download dataset to data folder')
            exit(0)

        dataset_path = osp.join(self.split_root, self.dataset)
        valid_splits = self.SUPPORTED_DATASETS[self.dataset]['splits']

        if self.lstm:
            self.corpus = Corpus()
            corpus_path = osp.join(dataset_path, 'corpus.pth')
            self.corpus = torch.load(corpus_path)

        if split not in valid_splits:
            raise ValueError(
                'Dataset {0} does not have split {1}'.format(
                    self.dataset, split))

        splits = [split]
        if self.dataset != 'referit':
            splits = ['train', 'val'] if split == 'trainval' else [split]
        for split in splits:
            imgset_file = '{0}_{1}.pth'.format(self.dataset, split)
            imgset_path = osp.join(dataset_path, imgset_file)
            self.images += torch.load(imgset_path)

        st_json_path = osp.join(self.data_root + '/ocr/google_ocr_results_reftext_rank_aggr.json')
        with open(st_json_path, "r") as load_f:
            self.st_json_data = json.load(load_f)


    def exists_dataset(self):
        return osp.exists(osp.join(self.split_root, self.dataset))

    def pull_item(self, idx):
        if self.dataset == 'flickr':
            img_file, bbox, phrase = self.images[idx]
        else:
            img_file, sampleid, bbox, phrase, attri = self.images[idx]
        ## box format: to x1y1x2y2
        if not (self.dataset == 'referit' or self.dataset == 'flickr'):
            bbox = np.array(bbox, dtype=int)
            bbox[2], bbox[3] = bbox[0]+bbox[2], bbox[1]+bbox[3]
        else:
            bbox = np.array(bbox, dtype=int)

        img_path = osp.join(self.im_dir, img_file)
        img = cv2.imread(img_path)
        ## duplicate channel if gray image
        if img.shape[-1] > 1:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img = np.stack([img] * 3)
        
        phrase_st = self.st_json_data[sampleid]

        return img, phrase, bbox, phrase_st

    def tokenize_phrase(self, phrase):
        return self.corpus.tokenize(phrase, self.query_len)

    def untokenize_word_vector(self, words):
        return self.corpus.dictionary[words]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img, phrase, bbox, st_list = self.pull_item(idx)
        phrase = phrase.lower()
        sent = phrase

        ## data loader for scene text
        st_num = 20  # M, the number of scene texts
        st_box_num = 40
        st_sent_length = 50
        st_list_word = []
        st_list_bbox = [np.array([0, 0, 0, 0], dtype=int)] * st_box_num
        st_list_bbox2word = [0] * st_box_num
        cnt = 0
        for ii in range(len(st_list)):
            if ii == st_num:
                break
            st_list_word.append(st_list[ii][0])
            for jj in st_list[ii][1]:
                if cnt == st_box_num:
                    break
                st_list_bbox[cnt] = np.array(jj, dtype=int)
                st_list_bbox[cnt][2], st_list_bbox[cnt][3] = st_list_bbox[cnt][0] + st_list_bbox[cnt][2], st_list_bbox[cnt][1] + st_list_bbox[cnt][3]
                st_list_bbox2word[cnt] = ii+1
                cnt += 1
        # To facilitate calculation, multiple scene texts are concatenated into a sentence
        st_sent = ' '.join(st_list_word)

        if self.augment:
            augment_flip, augment_hsv, augment_affine = True,True,True

        ## seems a bug in torch transformation resize, so separate in advance
        h,w = img.shape[0], img.shape[1]
        if self.augment:
            ## random horizontal flip
            if augment_flip and random.random() > 0.5:
                img = cv2.flip(img, 1)
                bbox[0], bbox[2] = w-bbox[2]-1, w-bbox[0]-1
                for ii in range(len(st_list_bbox2word)):
                    st_list_bbox[ii][0], st_list_bbox[ii][2] = w - st_list_bbox[ii][2] - 1, w - st_list_bbox[ii][0] - 1
                phrase = phrase.replace('right','*&^special^&*').replace('left','right').replace('*&^special^&*','left')
            ## random intensity, saturation change
            if augment_hsv:
                fraction = 0.50
                img_hsv = cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_RGB2BGR), cv2.COLOR_BGR2HSV)
                S = img_hsv[:, :, 1].astype(np.float32)
                V = img_hsv[:, :, 2].astype(np.float32)
                a = (random.random() * 2 - 1) * fraction + 1
                if a > 1:
                    np.clip(S, a_min=0, a_max=255, out=S)
                a = (random.random() * 2 - 1) * fraction + 1
                V *= a
                if a > 1:
                    np.clip(V, a_min=0, a_max=255, out=V)

                img_hsv[:, :, 1] = S.astype(np.uint8)
                img_hsv[:, :, 2] = V.astype(np.uint8)
                img = cv2.cvtColor(cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR), cv2.COLOR_BGR2RGB)
            img, _, ratio, dw, dh = letterbox(img, None, self.imsize)
            bbox[0], bbox[2] = bbox[0]*ratio+dw, bbox[2]*ratio+dw
            bbox[1], bbox[3] = bbox[1]*ratio+dh, bbox[3]*ratio+dh
            for ii in range(len(st_list_bbox2word)):
                st_list_bbox[ii][0], st_list_bbox[ii][2] = st_list_bbox[ii][0] * ratio + dw, st_list_bbox[ii][2] * ratio + dw
                st_list_bbox[ii][1], st_list_bbox[ii][3] = st_list_bbox[ii][1] * ratio + dh, st_list_bbox[ii][3] * ratio + dh
            ## random affine transformation
            if augment_affine:
                img, _, bbox, st_list_bbox, M = random_affine2(img, None, bbox, st_list_bbox, len(st_list_bbox2word), degrees=(-5, 5), translate=(0.10, 0.10), scale=(0.90, 1.10))
        else:   ## should be inference, or specified training
            img, _, ratio, dw, dh = letterbox(img, None, self.imsize)
            bbox[0], bbox[2] = bbox[0]*ratio+dw, bbox[2]*ratio+dw
            bbox[1], bbox[3] = bbox[1]*ratio+dh, bbox[3]*ratio+dh
            for ii in range(len(st_list_bbox2word)):
                st_list_bbox[ii][0], st_list_bbox[ii][2] = st_list_bbox[ii][0] * ratio + dw, st_list_bbox[ii][2] * ratio + dw
                st_list_bbox[ii][1], st_list_bbox[ii][3] = st_list_bbox[ii][1] * ratio + dh, st_list_bbox[ii][3] * ratio + dh


        ## Norm, to tensor
        if self.transform is not None:
            img = self.transform(img)
        if self.lstm:
            phrase = self.tokenize_phrase(phrase)
            word_id = phrase
            word_mask = np.array(word_id>0,dtype=int)
        else:
            ## encode phrase to bert input
            examples = read_examples(phrase, idx)
            features = convert_examples_to_features(
                examples=examples, seq_length=self.query_len, tokenizer=self.tokenizer)
            word_id = features[0].input_ids
            word_mask = features[0].input_mask
            ## for scene text
            word_id_st_list = [[0] * 20 for _ in range(st_num)]
            word_mask_st_list = [[0] * 20 for _ in range(st_num)]
            for ii in range(len(st_list_word)):
                examples_st_word = read_examples(st_list_word[ii], idx)
                features_st_word = convert_examples_to_features(examples=examples_st_word, seq_length=self.query_len, tokenizer=self.tokenizer)
                word_id_st_list[ii] = features_st_word[0].input_ids
                word_mask_st_list[ii] = features_st_word[0].input_mask
            ## for the 'sentence' composed of multiple scene texts
            examples_st_sent = read_examples(st_sent, idx)
            features_st_sent = convert_examples_to_features(examples=examples_st_sent, seq_length=st_sent_length, tokenizer=self.tokenizer)
            word_id_st_sent = features_st_sent[0].input_ids
            word_mask_st_sent = features_st_sent[0].input_mask
            word_id_st_sent2word = [0] * st_sent_length
            word_id_st_sent2word[0] = -1
            word_st_position_cnt = 1
            for ii in range(len(st_list_word)):
                for jj in range(len(np.nonzero(word_id_st_list[ii])[0])):
                    if word_id_st_list[ii][jj] == 101 or word_id_st_list[ii][jj] == 102:
                        continue
                    if word_st_position_cnt == (st_sent_length - 1):
                        break
                    word_id_st_sent2word[word_st_position_cnt] = ii
                    word_st_position_cnt += 1
                if word_st_position_cnt == (st_sent_length - 1):
                    break
            word_id_st_sent2word[word_st_position_cnt] = -2
        if self.testmode:
            return img, (sent,np.array(word_id, dtype=int)), np.array(word_mask, dtype=int), \
                np.array(bbox, dtype=np.float32), np.array(ratio, dtype=np.float32), \
                np.array(dw, dtype=np.float32), np.array(dh, dtype=np.float32), self.images[idx][0], np.array(word_id_st_sent2word, dtype=int), np.array(st_list_bbox, dtype=np.float32), np.array(word_id_st_sent, dtype=int), np.array(word_mask_st_sent, dtype=int), np.array(st_list_bbox2word, dtype=int)
        else:
            return img, np.array(word_id, dtype=int), np.array(word_mask, dtype=int), \
            np.array(bbox, dtype=np.float32), np.array(word_id_st_sent2word, dtype=int), np.array(st_list_bbox, dtype=np.float32), np.array(word_id_st_sent, dtype=int), np.array(word_mask_st_sent, dtype=int), np.array(st_list_bbox2word, dtype=int)
