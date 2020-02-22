#!/usr/bin/env python
# coding: utf-8
import os
import argparse
import numpy as np
import random
import pathlib

from itertools import accumulate
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, Subset, TensorDataset


from ax.plot.contour import plot_contour
from ax.service.managed_loop import optimize
from ax.utils.notebook.plotting import render


def SamplingData(pictures, labels, BatchSize):
    """ Sampling the data into training, validation and test set """
    N = len(labels)
    
    TrainIndex = int(2/3*N)
    TestIndex  = int(1/6*N + TrainIndex)
    
    PicturesTrain = pictures[0:TrainIndex]
    LabelsTrain = labels[0:TrainIndex]
    
    PicturesTest = pictures[TrainIndex:TestIndex]
    LabelsTest   = labels[TrainIndex:TestIndex]
    
    PicturesVal = pictures[TestIndex:]
    LabelsVal = labels[TestIndex:]
    
    Train_dataset = TensorDataset(PicturesTrain, LabelsTrain)
    Test_dataset = TensorDataset(PicturesTest, LabelsTest)
    Val_dataset = TensorDataset(PicturesVal, LabelsVal)
    
    Train = DataLoader( Train_dataset, batch_size=BatchSize, drop_last = False, shuffle = True)
    Test = DataLoader( Test_dataset, batch_size=BatchSize * 2, drop_last = False, shuffle = False)
    Val = DataLoader( Val_dataset, batch_size=BatchSize * 2, drop_last = False, shuffle = False)
    
    Data = {}
    Data["Train"] = Train
    Data["Test"] = Test
    Data["Val"] = Val
    
    return Data


class NET(nn.Module):
    """
        Convolutional Neural Network.
        """
    
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 20, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(27 * 27 * 20, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 27 * 27 * 20)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=-1)




def train(net: torch.nn.Module,
          train_loader: DataLoader,
          parameters: Dict[str, float],
          dtype: torch.dtype,
          device: torch.device,
          ) -> nn.Module:

    # train(net=NET(),train_loader=combined_train_valid_loader,parameters=best_arm.parameters,dtype=dtype,device=device,):
    """
        Train CNN on provided data set.
        
        Args:
        net: initialized neural network
        train_loader: DataLoader containing training set
        parameters: dictionary containing parameters to be passed to the optimizer.
        - lr: default (0.001)
        - momentum: default (0.0)
        - weight_decay: default (0.0)
        - num_epochs: default (1)
        dtype: torch dtype
        device: torch device
        Returns:
        nn.Module: trained CNN.
        """
    # Initialize network
    net.to(dtype=dtype, device=device)  # pyre-ignore [28]
    net.train()
    # Define loss and optimizer
    criterion = nn.NLLLoss(reduction="sum")
    optimizer = optim.SGD(net.parameters(), lr=parameters.get("lr", 0.001), momentum=parameters.get("momentum", 0.0), weight_decay=parameters.get("weight_decay", 0.0),)
    scheduler = optim.lr_scheduler.StepLR(optimizer,step_size=int(parameters.get("step_size", 30)),gamma=parameters.get("gamma", 1.0),)
    num_epochs = parameters.get("num_epochs", 1)
    
    # Train Network
    for _ in range(num_epochs):
        for inputs, labels in train_loader:
            # move data to proper dtype and device
            inputs = inputs.to(dtype=dtype, device=device)
            labels = labels.to(device=device)
            
            # zero the parameter gradients
            optimizer.zero_grad()
            
            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
    return net


def evaluate(NET: nn.Module, data_loader: DataLoader, dtype: torch.dtype, device: torch.device
             ) -> float:
             
             
             #net=NET(),data_loader=TestSet,dtype=dtype,device=device,):
    """
        Compute classification accuracy on provided dataset.
        
        Args:
        net: trained model
        data_loader: DataLoader containing the evaluation set
        dtype: torch dtype
        device: torch device
        Returns:
        float: classification accuracy
        """
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in data_loader:
            # move data to proper dtype and device
            inputs = inputs.to(dtype=dtype, device=device)
            labels = labels.to(device=device)
            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return correct / total


    
def test_data_set(data):
    """ This test the data if it torch or not """
    if "torch" not in str(type(data)):
        raise ValueError('Make sure the data is type torch')
    else:
        pass


# Setting up a search space
def SetUpTheSearchSpace(args):
    """ This fucntion is suppose to make the searching space. """
    parameters=[
                {"name": "lr", "type": "range", "bounds": [float(args["Llr"]) , float(args["Hlr"])], "log_scale": False},
                #{"name": "weight_decay", "type": "range", "bounds": [float(args["LWD"]) ,float( args["HWD"]) ] },
                {"name": "momentum", "type": "range", "bounds": [args["LMM"] , args["HMM"]]},
                ]
    return parameters


def plottingConture(model):
    """Plotting conture of the search space to see what gives the best results"""
    return render(plot_contour(model=model, param_x='lr', param_y='momentum', metric_name='accuracy'))


def train_evaluate(NET,parameterization,train_loader):
    " Training the network"
    net = NET
    net = train( net=net, train_loader=train_loader, parameters=parameterization, dtype=torch.float16,device="cpu")
    return evaluate( net=net, data_loader=valid_loader,dtype=torch.float16,device="cpu",)
    
    
def optimizeNetwork(NET,parameters,train_loader):
    """ Finding the best parameters for the network and saving them """
    best_parameter, value, experiment, model = optimize( parameters, evaluation_function=train_evaluate(NET,parameters,train_loader), objective_name='accuracy',)
    
    Parameters.append(best_parameters)
    values.append(values)
    Experiments.append(experiment)
    
    return best_parameter, value, experiment, model




