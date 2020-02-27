#!/usr/bin/env python
# coding: utf-8

import Project_CSME802_2
import numpy as np
import torch
import unittest

class best_arm1:
        parameters = {}
        parameters["lr"] = 0.001
        parameters["momentum"] = 0.001
        parameters["weight_decay"] = 0.001

class TestCode(unittest.TestCase):

    def test_load_data(self):
        
        data = Project_CSME802_2.load_data()
        
        self.assertEqual( data['X_train'].shape , (50000, 1, 60, 60) )
        self.assertEqual( data['X_valid'].shape , (10000, 1, 60, 60) )
        self.assertEqual( data['X_test'].shape , (10000, 1, 60, 60) )

    
    
    def test_eval_BO(self):
        
        net  = Project_CSME802_2.Net()
        data = Project_CSME802_2.load_data()
        accuracy = Project_CSME802_2.eval_BO(net=net, Input=data['X_valid'], Label=data['y_valid'],)
        
        self.assertEqual( type(accuracy), np.float64 )
        self.assertEqual( accuracy.shape, () )

    
    def test_train_epoch(self):
        
        net  = Project_CSME802_2.Net()
        data = Project_CSME802_2.load_data()
        best_arm = best_arm1()
        cost_mean, accuracy = Project_CSME802_2.train_epoch(net=net, Input=data['X_train'],Label=data['y_train'], parameters=best_arm.parameters,)
        
        self.assertEqual( type(accuracy), np.float64 )
        self.assertEqual( type(cost_mean), np.float32 )
        self.assertEqual( accuracy.shape, () )
        self.assertEqual( cost_mean.shape, () )


    def test_eval_epoch(self):
        
        net  = Project_CSME802_2.Net()
        data = Project_CSME802_2.load_data()
        accuracy = Project_CSME802_2.eval_epoch( net=net, Input=data['X_valid'], Label=data['y_valid'],)
        
        self.assertEqual( type(accuracy), np.float64 )
        self.assertEqual( accuracy.shape, () )


if __name__ == "__main__":
    unittest.main()
