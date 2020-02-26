#!/usr/bin/env python
# coding: utf-8

import numpy as np
import os

def test_load_data():
    '''  '''
    if not os.path.isfile(filename):
        raise ValueError("Input file does not exist: {0}. I'll quit now.".format(filename))

    # code to load and parse the data from input file
    data = np.loadtxt(filename, delimiter=',')

    if len(data) < 5:
        # there should be 5 rows
        raise ValueError("Not enough rows in input file.")
    
    return data
