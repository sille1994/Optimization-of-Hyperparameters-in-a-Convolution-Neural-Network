#!/usr/bin/env python
# coding: utf-8
import os
import argparse
import numpy as np
import random

#from ax.plot.contour import plot_contour
#from ax.service.managed_loop import optimize
#from ax.utils.notebook.plotting import render
from botorch.models import SingleTaskGP
from gpytorch.constraints import GreaterThan
from gpytorch.mlls import ExactMarginalLogLikelihood

from torch.optim import SGD, Adam
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset
import torchvision




# Set values for the seach space
#parser = argparse.ArgumentParser(description='Set values for the search space for the hyperparameters')
#parser.add_argument("-Llr", default=10^(-5), type=float, help='Lower bound of the learning rate')
#parser.add_argument("-Hlr", default=10^(-1), type=float, help='Higher bound of the learning rate')
#parser.add_argument("-WD" , default= False, type=float, help='Is weight decay used in the optimize algorithm?')
#parser.add_argument("-LWD", default=4e-4, type=float, help='Lower bound of the weight decay')
#parser.add_argument("-HWD", default=4e-2, type=float, help='Higher bound of the weight decay')
#parser.add_argument("-MM" , default= False, type=float, help='Is momentum used in the optimize algorithm?')
#parser.add_argument("-LMM", default=0.5, type=float, help='Lower bound of the momentum')
#parser.add_argument("-HMM", default=0.9, type=float, help='Higher bound of the momentum')
#parser.add_argument("-Opt", default='SGD', type=str, help='Choose the optimizer: Stochastic gradient descent (SGD) and Adam optimization algorithm (Adam)')
#args = parser.parse_args()
args = {}
args["Llr"] =10^(-5)
args["Hlr"] =10^(-1)
args["LWD"] =4e-4
args["HWD"] =4e-2
args["LMM"] =0.5
args["HMM"] =0.9



search_space = 0
data = 0

# Load data
# Source: https://www.codementor.io/@dejanbatanjac/pytorch-the-missing-manual-on-loading-mnist-dataset-wjeh5top7
data = DataLoader(torchvision.datasets.MNIST("/data/mnist", train=True, download=True))
    
pictures = data.dataset.datase
pictures = tensor.to(dtype=torch.float16)
pictures = tensor.reshape(tensor.size(0),-1)

labels = data.dataset.targets
labels = labels.to(dtype=torch.long)



def SamplingData(pictures,labels, BatchSize):
    """ Sampling the data into training, validation and test set """
    N = labels
    
    TrainIndex = int(2/3*N)
    TestIndex  = int(1/6*N + TrainIndex)
    
    PicturesTrain = pictures[0:TrainIndex]
    LabelsTrain = labels[0:TrainIndex]
    
    PicturesTest = pictures[TrainIndex:TestIndex]
    LabelsTest   = labels[TrainIndex:TestIndex]
    
    PicturesVal = pictures[TestIndex:end]
    LabelsVal = labels[TestIndex:end]
    
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
                {"name": "lr", "type": "range", "bounds": [args.Llr, args.Hlr]},
                {"name": "weight", "type": "range", "bounds": [args.LWD, args.HWD] },
                {"name": "momentum", "type": "range", "bounds": [args.LMM, args.HMM]},
                ],
    return parameters

def train_evaluate(parameterization):
    net = NET()
    net = train(net=net, train_loader=train_loader, parameters=parameterization, dtype=dtype, device=device)
    return evaluate(net=net,data_loader=valid_loader,dtype=dtype,device=device,)
    
    
def optimizeNetwork(parameters,train_evaluate):
    """ Finding the best parameters for the network"""
    best_parameters, values, experiment, model = optimize(parameters, evaluation_function=train_evaluate,objective_name='accuracy',)
    return best_parameters, values, experiment, model


def plottingConture(model):
    """Plotting conture of the search space to see what gives the best results"""
    return render(plot_contour(model=model, param_x='lr', param_y='momentum', metric_name='accuracy'))



def BO(Ypred,Ytruth):
    """ Calculating the surrogate function and the aqcuisition function """

    # Initialize the Gaussian Likelihood
    BOmodel = SingleTaskGP(train_X=Ypred, train_Y=Ytruth)
    BOmodel.likelihood.noise_covar.register_constraint("raw_noise", GreaterThan(1e-5))

    # The marginal log likelihood is defined
    mll = ExactMarginalLogLikelihood(likelihood=BOmodel.likelihood, model=BOmodel)
    mll = mll.to(Ypred)

    #        if args.optimizer == 'SGD':
    #            optimizer = SGD([{'params': model.parameters()}], args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    #        if args.optimizer == 'Adam':
    #            optimizer = Adam([{'params': model.parameters()}], args.lr, betas=(0.9, 0.999), weight_decay=args.weight_decay, amsgrad=True)

    return BOmodel, mll




