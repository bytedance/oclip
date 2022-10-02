# *************************************************************************
# This file may have been modified by Bytedance Inc. (“Bytedance Inc.'s Mo-
# difications”). All Bytedance Inc.'s Modifications are Copyright (2022) B-
# ytedance Inc..  
# *************************************************************************

import os
import sys
import math
import logging
import functools
import random
import pdb
import json
import pickle

import pandas as pd
import numpy as np
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from typing import Union
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from torch.utils.data.distributed import DistributedSampler
import torchvision.datasets as datasets
# from webdataset.utils import identity
# import webdataset as wds
import csv

# from clip.clip import tokenize

class CsvDataset(Dataset):
    def __init__(self, input_filename, transforms, first_stage, img_key, caption_key, char_dict_pth, sep="\t", 
                 single_text=False, text_batch_size=256, vocab_size=40000, context_length=77, image_resolution=512):
        logging.debug(f'Loading csv data from {input_filename}.')
        df = pd.read_csv(input_filename, sep=sep, quoting=csv.QUOTE_NONE)

        self.images = df[img_key].tolist()
        self.captions = df[caption_key].tolist()
        self.transforms = transforms
        self.single_text = single_text
        self.vocab_size = vocab_size
        self.text_batch_size = text_batch_size
        with open(char_dict_pth, 'rb') as f:
            self.letters = pickle.load(f)
            self.letters = [chr(x) for x in self.letters]
        self.p2idx = {p: idx+1 for idx, p in enumerate(self.letters)}
        self.idx2p = {idx+1: p for idx, p in enumerate(self.letters)}

        self.idx_mask = len(self.letters) + 1
        self.EOS = len(self.letters) + 2
        self.image_resolution = image_resolution
        self.first_stage = first_stage

        self.max_len = 32
        self.word_len = 25

        self.context_length = context_length

        logging.debug('Done loading data.')

    def tokenize(self, text):
        token = torch.zeros(self.word_len)
        for i in range(min(len(text), self.word_len)):
            token[i] = self.p2idx[text[i]]
        if len(text) >= self.word_len:
            token[-1] = self.EOS
        else:
            token[len(text)] = self.EOS

        return token

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        images = self.transforms(Image.open(str(self.images[idx])))
        all_texts = self.captions[idx].split(' ')
        texts = torch.zeros((self.max_len, self.word_len))
        masked_chars = torch.zeros(self.max_len)
        for i in range(min(len(all_texts), self.max_len)):
            t = self.tokenize(all_texts[i])
            if not self.first_stage:
                rand_idx = random.randint(0, min(len(all_texts[i]), self.word_len) - 1)
                masked_chars[i] = t[rand_idx].clone()
                t[rand_idx] = self.idx_mask
            texts[i] += t

        # image masks can be used to mask out the padding regions during training
        image_masks = torch.zeros((self.image_resolution // 32, self.image_resolution // 32), dtype=torch.bool)

        if self.first_stage:
            return images, texts.long(), image_masks
        else:        
            return images, texts.long(), masked_chars.long(), image_masks
       
@dataclass
class DataInfo:
    dataloader: DataLoader
    sampler: DistributedSampler


def get_csv_dataset(args, preprocess_fn, is_train):
    input_filename = args.train_data if is_train else args.val_data
    assert input_filename
    dataset = CsvDataset(
        input_filename,
        preprocess_fn,
        first_stage=args.first,
        img_key=args.csv_img_key,
        caption_key=args.csv_caption_key,
        char_dict_pth=args.char_dict_pth,
        sep=args.csv_separator)
    num_samples = len(dataset)
    sampler = DistributedSampler(dataset) if args.distributed and is_train else None
    shuffle = is_train and sampler is None

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=is_train,
    )
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)

    return DataInfo(dataloader, sampler)

    
def get_data(args, preprocess_fns):
    preprocess_train, preprocess_val = preprocess_fns
    data = {}

    if args.train_data:
        data["train"] = get_csv_dataset(args, preprocess_train, is_train=True)

    return data
