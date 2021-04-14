import os
import sys
import glob
import random
import math
import logging

import numpy as np
from PIL import Image
import torch
import torch.utils.data as data
import torchvision.datasets as datasets
from torchvision.transforms import functional
import torchvision.transforms as transforms
from torch.utils.data.dataloader import default_collate
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler, SubsetRandomSampler
from torchvision.transforms.functional import to_tensor


def make_datasets(valid_size=0.2,test_size=0.1,num_workers=4):
    train_transforms,validation_transforms,test_transforms = None, None, None
    train_data = None

    # obtain training indices that will be used for validation
    num_train = len(train_data)
    indices = list(range(num_train))
    np.random.shuffle(indices)
    valid_split = int(np.floor((valid_size) * num_train))
    test_split = int(np.floor((valid_size+test_size) * num_train))
    valid_idx, test_idx, train_idx = indices[:valid_split], indices[valid_split:test_split], indices[test_split:]

    # define samplers for obtaining training and validation batches
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)
    test_sampler = SubsetRandomSampler(test_idx)

    # prepare data loaders (combine dataset and sampler)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=32,
        sampler=train_sampler, num_workers=num_workers)
    valid_loader = torch.utils.data.DataLoader(train_data, batch_size=32, 
        sampler=valid_sampler, num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(train_data, batch_size=1, 
        sampler=test_sampler, num_workers=num_workers)
    
    return train_loader,valid_loader,test_loader

    