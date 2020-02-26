#!/usr/bin/env python
# coding: utf-8

import Project_CSME802
import pytest
import numpy as np
import does
import torch

def test_load_data():
    
    # test if function returns a dict with train, test and valid dataset.
    data = Project_CSME802.load_data()
    assert type(data) == dict
    assert train in data or Train in data
    assert test in data or Test in data
    assert valid in data or valid in data


def test_Net(DIM = 60):
    
    # test forward pass on dummy data
    net = Project_CSME802.Net()
    x = np.random.normal(0,1, (45, 1, DIM, DIM)).astype('float32')
    x = Variable(torch.from_numpy(x))
    if torch.cuda.is_available():
        x = x.cuda()
    try:
        output = net(x)
    except:
        print("The data is not the right diemension for the network!")
