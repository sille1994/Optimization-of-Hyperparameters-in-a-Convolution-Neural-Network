""" This code is used to test the script: project_csme802_2_2.py. """

#!/usr/bin/env python
# coding: utf-8

import unittest
import numpy as np
import project_csme802


class Parameters:
    """ Parameters used to test the code """
    parameters = {}
    parameters["lr"] = 0.001
    parameters["momentum"] = 0.001
    parameters["weight_decay"] = 0.001


class TestCode(unittest.TestCase):
    """ Testing the code """
    def test_load_data(self):
        """ Testing the load_data-function """
        data = project_csme802.load_data()

        self.assertEqual(data['x_train'].shape, (50000, 1, 60, 60))
        self.assertEqual(data['x_valid'].shape, (10000, 1, 60, 60))
        self.assertEqual(data['x_test'].shape, (10000, 1, 60, 60))

    def test_eval_bayesian_optimization(self):
        """ Testing the eval_bayesian_optimization-function """
        net = project_csme802.Net()
        data = project_csme802.load_data()
        accuracy = project_csme802.eval_bayesian_optimization(
            net=net, input_picture=data['x_valid'], label_picture=data['y_valid'],)

        self.assertEqual(type(accuracy), np.float64)
        self.assertEqual(accuracy.shape, ())

    def test_train_epoch(self):
        """ Testing the train_epoch-function """
        net = project_csme802.Net()
        data = project_csme802.load_data()
        best_arm = Parameters()
        cost_mean, accuracy = project_csme802.train_epoch(net=net,\
            input_picture=data['X_train'], label_picture=data['y_train'],\
            parameters=best_arm.parameters,)

        self.assertEqual(type(accuracy), np.float64)
        self.assertEqual(type(cost_mean), np.float32)
        self.assertEqual(accuracy.shape, ())
        self.assertEqual(cost_mean.shape, ())

    def test_eval_epoch(self):
        """ Testing the eval_epoch-function """
        net = project_csme802.Net()
        data = project_csme802.load_data()
        accuracy = project_csme802.eval_epoch(
            net=net, input_picture=data['X_valid'], label_picture=data['y_valid'],)

        self.assertEqual(type(accuracy), np.float64)
        self.assertEqual(accuracy.shape, ())


if __name__ == "__main__":
    unittest.main()
