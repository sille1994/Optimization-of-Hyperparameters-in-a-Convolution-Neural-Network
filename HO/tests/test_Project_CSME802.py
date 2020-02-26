#!/usr/bin/env python
# coding: utf-8

import Project_CSME802
import pytest
import numpy as np
import does

def test_load_data():
    
    # test if function returns a dict with train, test and valid dataset.
    data = Project_CSME802.load_data()
    assert type(data) == dict
    assert train in data or Train in data
    assert test in data or Test in data
    assert valid in data or valid in data
