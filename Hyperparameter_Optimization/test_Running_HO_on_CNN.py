""" This code is used to test the script: project_csme802_2_2.py. """

#!/usr/bin/env python
# coding: utf-8

import unittest
import numpy as np
from typing import Dict, List
import Running_HO_on_CNN_unittest


class Parameters:
    """ Parameters used to test the code """
    parameters = {}
    parameters["lr"] = 0.001
    parameters["momentum"] = 0.001
    parameters["weight_decay"] = 0.001
    parameters["step_size"] = 20
    parameters["gamma"] = 1.0
    parameters["num_epochs"] = 1

class TestCode(unittest.TestCase):
    """ Testing the code """
    def test_load_data(self):
        """ Testing the load_data-function """
        data = Running_HO_on_CNN_unittest.load_data()
        self.assertEqual(data['x_train'].shape, (50000, 1, 60, 60))
        self.assertEqual(data['x_valid'].shape, (10000, 1, 60, 60))
        self.assertEqual(data['x_test'].shape, (10000, 1, 60, 60))

    def test_eval_bayesian_optimization(self):
        """ Testing the eval_bayesian_optimization-function """
        net = Running_HO_on_CNN_unittest.Net()
        data = Running_HO_on_CNN_unittest.load_data()
        accuracy = Running_HO_on_CNN_unittest.eval_bayesian_optimization(net=net,\
                    input_picture=data['x_valid'], label_picture=data['y_valid'],)
        self.assertEqual((type(accuracy)), float)

    def test_train_bayesian_optimization(self):
        """ Testing the train_epoch-function """
        net = Running_HO_on_CNN_unittest.Net()
        data = Running_HO_on_CNN_unittest.load_data()
        best_arm = Parameters()
        _, cost_mean, accuracy = Running_HO_on_CNN_unittest.train_bayesian_optimization(net=net,\
                                            input_picture=data['x_train'],\
                                            label_picture=data['y_train'],\
                                            parameters=best_arm.parameters,)
        self.assertEqual(type(accuracy), np.float64)
        self.assertEqual(type(cost_mean), np.float32)
        self.assertEqual(accuracy.shape, ())
        self.assertEqual(cost_mean.shape, ())


if __name__ == "__main__":
    unittest.main()
