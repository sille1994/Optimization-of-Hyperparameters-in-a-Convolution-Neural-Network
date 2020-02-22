#!/usr/bin/env python
# coding: utf-8
import os
import argparse
import numpy as np
import random
import pathlib

from ax.plot.contour import plot_contour
from ax.service.managed_loop import optimize
from ax.utils.notebook.plotting import render
from botorch.models import SingleTaskGP
from gpytorch.constraints import GreaterThan
from gpytorch.mlls import ExactMarginalLogLikelihood

from torch.optim import SGD, Adam
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset
import torchvision
from torch.autograd import Variable


from functions import SamplingData, NET, train, evaluate, SetUpTheSearchSpace, train_evaluate, optimizeNetwork

torch.manual_seed(12345)
dtype = torch.float
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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


file = pathlib.Path("./data/")
if file.exists ():
    data = DataLoader(torchvision.datasets.MNIST("./data/mnist", train=True, download=False))
else:
    data = DataLoader(torchvision.datasets.MNIST("./data/mnist", train=True, download=True))
    
pictures = data.dataset.data
pictures = pictures.to(dtype=torch.float16)
print(pictures[0].shape)

labels = data.dataset.targets
print(labels,type(labels))

labels = labels.to(dtype=torch.long)
Parameters = []
values = []
Experiments = []


NUM_EPOCHS = 10
BATCH_SIZE = 256
DIM = pictures[0].shape[0]
NUM_CLASSES = 10

Data = SamplingData(pictures,labels, BATCH_SIZE)
TrainSet = Data["Train"]
TestSet  = Data["Test"]
ValSet   = Data["Val"]

print(NET())
net = NET()


i = 0
while i < NUM_EPOCHS:
    i = i + 1
    parameters = SetUpTheSearchSpace(args)
    best_parameter, value, experiment, model = optimizeNetwork(net,parameters,ValSet)
    data = experiment.fetch_data()
    df = data.df
    best_arm_name = df.arm_name[df['mean'] == df['mean'].max()].values[0]
    best_arm = experiment.arms_by_name[best_arm_name]
    net = train(net=net,train_loader=TestSet,parameters=best_arm.parameters,dtype=torch.float16,device="cpu",)

    test_accuracy = evaluate(net=net,data_loader=TestSet,dtype=torch.float16,device="cpu",)
    print(f"Classification Accuracy (test set): {round(test_accuracy*100, 2)}%")

#def BO(Ypred,Ytruth):
#    """ Calculating the surrogate function and the aqcuisition function """

    # Initialize the Gaussian Likelihood
#    BOmodel = SingleTaskGP(train_X=Ypred, train_Y=Ytruth)
#    BOmodel.likelihood.noise_covar.register_constraint("raw_noise", GreaterThan(1e-5))

    # The marginal log likelihood is defined
#    mll = ExactMarginalLogLikelihood(likelihood=BOmodel.likelihood, model=BOmodel)
#    mll = mll.to(Ypred)

    #        if args.optimizer == 'SGD':
    #            optimizer = SGD([{'params': model.parameters()}], args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    #        if args.optimizer == 'Adam':
    #            optimizer = Adam([{'params': model.parameters()}], args.lr, betas=(0.9, 0.999), weight_decay=args.weight_decay, amsgrad=True)

#    return BOmodel, mll




