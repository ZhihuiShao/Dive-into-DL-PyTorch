import collections
import math
import os
import random
import sys
import tarfile
import time
import json
import zipfile
from tqdm import tqdm
from PIL import Image
from collections import namedtuple

from IPython import display
from matplotlib import pyplot as plt
import torch
from torch import nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torchtext
import torchtext.vocab as Vocab
import numpy as np

def load_text_from_zip(zip_file_name,cap_len = 10000):
    """read chars and generate corpus from zip"""
    with zipfile.ZipFile(zip_file_name) as zin:
        with zin.open('jaychou_lyrics.txt') as f:
            corpus_chars = f.read().decode('utf-8')
    corpus_chars = corpus_chars.replace('\n',' ').replace('\r', ' ')
    idx_to_char = list(set(corpus_chars))
    char_to_idx = dict((c,i) for i,c in enumerate(idx_to_char))
    vocab_size = len(idx_to_char)
    corpus_ints = [char_to_idx[c] for c in corpus_chars]
    return corpus_chars[:cap_len], idx_to_char, char_to_idx, corpus_ints

def data_iter_random(corpus_ints,batch_size,num_steps,device='cuda'):
    """
    yeild random chars from corpusl
    each sample include strings with len=num_steps for both X and Y, Y is right shift 1 of X
    """
    # number of possible sentence
    num_examples = (len(corpus_ints)-1)//num_steps #because we need Y available, hence -1
    epoch_size = num_examples // batch_size # each batch with batch_size*num_steps chars
    example_indices = list(range(num_examples)) # which sentence
    random.shuffle(example_indices) # to generate random index
    for i in range(epoch_size):
        start = i*batch_size
        end = start+batch_size
        sentence_index = example_indices[start:end] # index is sentence index
        X = [corpus_ints[j*num_steps:(j+1)*num_steps] for j in sentence_index]
        Y = [corpus_ints[(j*num_steps+1):((j+1)*num_steps+1)] for j in sentence_index]

        yield torch.tensor(X,dtype=torch.float32,device=device),torch.tensor(Y,dtype=torch.float32,device=device)

def data_iter(corpus_ints,batch_size,num_steps,device='cuda'):
    num_examples = (len(corpus_ints)-1)//num_steps #because we need Y available, hence -1
    epoch_size = num_examples // batch_size # each batch with batch_size*num_steps chars
    example_indices = list(range(num_examples)) # which sentence
    for i in range(epoch_size):
        start = i*batch_size
        end = start+batch_size
        sentence_index = example_indices[start:end] # index is sentence index
        X = [corpus_ints[j*num_steps:(j+1)*num_steps] for j in sentence_index]
        Y = [corpus_ints[(j*num_steps+1):((j+1)*num_steps+1)] for j in sentence_index]

        yield torch.tensor(X,dtype=torch.float32,device=device),torch.tensor(Y,dtype=torch.float32,device=device)