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


def load_data_fashion_mnist(batch_size,resize=None,root='~/Datasets/FashionMNIST'):
    "Download fashion minist dataset and generate data loader in pytorch"
    trans = [] #the transforms when loading data
    if resize: # resize is required
        trans.append(torchvision.transforms.Resize(size=resize))
    trans.append
