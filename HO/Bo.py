#!/usr/bin/env python
# coding: utf-8
import os
import argparse
import ax
from botorch.models import SingleTaskGP
from gpytorch.constraints import GreaterThan
from gpytorch.mlls import ExactMarginalLogLikelihood
from torch.optim import SGD, Adam

# Set values for the seach space
parser = argparse.ArgumentParser(description='Set values for the search space for the hyperparameters')
#parser.add_argument('-Llr', default=10^(-5), type=float, help='Lower bound of the learning rate')
#parser.add_argument('-Hlr', default=10^(-1), type=float, help='Higher bound of the learning rate')
#parser.add_argument('-WD'  , default= False, type=float, help='Is weight decay used in the optimize algorithm?')
#parser.add_argument('-LWD', default=4e-4, type=float, help='Lower bound of the weight decay')
#parser.add_argument('-HWD', default=4e-2, type=float, help='Higher bound of the weight decay')
#parser.add_argument('-MM'  , default= False, type=float, help='Is momentum used in the optimize algorithm?')
#parser.add_argument('-LMM', default=0.5, type=float, help='Lower bound of the momentum')
#parser.add_argument('-HMM', default=0.9, type=float, help='Higher bound of the momentum')
#parser.add_argument('-Opt' , default='SGD', type=str, help='Choose the optimizer: Stochastic gradient descent (SGD) and Adam optimization algorithm (Adam)')

args = 0
search_space = 0


# Setting up a search space

def SetUpTheSearchSpace(search_space,args):
    """ This fucntion is suppose to make the searching space. """

    return search_space


def BO(Ypred,Ytruth,args):
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




