#!/usr/bin/env python
# coding: utf-8

# 1. Loading


import warnings
from ax.utils.notebook.plotting import render, init_notebook_plotting
from ax.service.managed_loop import optimize
from ax.plot.trace import optimization_trace_single_method
from ax.plot.contour import plot_contour
from typing import Dict, List, Optional, Tuple
from itertools import accumulate
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import torchvision
import torch
import numpy as np
import matplotlib.pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'inline')
warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.functional")



#get_ipython().system('if [ ! -f mnist_cluttered_60x60_6distortions.npz ]; then wget -N https://www.dropbox.com/s/rvvo1vtjjrryr7e/mnist_cluttered_60x60_6distortions.npz; else echo "mnist_cluttered_60x60_6distortions.npz already downloaded"; fi')


# 2. Load data and minding to be smaller.


NUM_EPOCHS = 1
BATCH_SIZE = 256
DIM = 60
NUM_CLASSES = 10
mnist_cluttered = "mnist_cluttered_60x60_6distortions.npz"



def load_data():
    """
    Load the npz-map and sort the data in a train, valid and test datset.

    Args:
        None
    Returns:
        dict: A dictionary with the data sorted.
    """

    data = np.load(mnist_cluttered)
    X_train, y_train = data['x_train'], np.argmax(data['y_train'], axis=-1)
    X_valid, y_valid = data['x_valid'], np.argmax(data['y_valid'], axis=-1)
    X_test, y_test = data['x_test'], np.argmax(data['y_test'], axis=-1)

    # reshape for convolutions
    X_train = X_train.reshape((X_train.shape[0], 1, DIM, DIM))
    X_valid = X_valid.reshape((X_valid.shape[0], 1, DIM, DIM))
    X_test = X_test.reshape((X_test.shape[0], 1, DIM, DIM))

    print("Train samples:", X_train.shape)
    print("Validation samples:", X_valid.shape)
    print("Test samples:", X_test.shape)

    return dict(X_train=np.asarray(X_train, dtype='float32'), y_train=y_train.astype('int32'), X_valid=np.asarray(X_valid, dtype='float32'), y_valid=y_valid.astype('int32'), X_test=np.asarray(X_test, dtype='float32'), y_test=y_test.astype('int32'), num_examples_train=X_train.shape[0], num_examples_valid=X_valid.shape[0], num_examples_test=X_test.shape[0], input_height=X_train.shape[2], input_width=X_train.shape[3], output_dim=10,)


data = load_data()

#idx = 0
#canvas = np.zeros((DIM*NUM_CLASSES, NUM_CLASSES*DIM))
#for i in range(NUM_CLASSES):
#    for j in range(NUM_CLASSES):
#        canvas[i*DIM:(i+1)*DIM, j*DIM:(j+1) *
#               DIM] = data['X_train'][idx].reshape((DIM, DIM))
#        idx += 1
#plt.figure(figsize=(10, 10))
#plt.imshow(canvas, cmap='gray')
#plt.title('Cluttered handwritten digits')
#plt.axis('off')
#plt.show()


# 3. Defining the Convolutional Neural Network

class Net(nn.Module):

    def __init__(self, input_channels=1, input_height=60, input_width=60, num_classes=10, num_zoom=3):
        super(Net, self).__init__()
        self.input_channels = input_channels
        self.input_height = input_height
        self.input_width = input_width
        self.num_classes = num_classes
        self.num_zoom = num_zoom

        # Spatial transformer localization-network
        # nn.Sequential http://pytorch.org/docs/master/nn.html#torch.nn.Sequential
        #   A sequential container.
        #   Modules will be added to it in the order they are passed in the constructor.
        self.localization = nn.Sequential(nn.Conv2d(in_channels=input_channels, out_channels=8, kernel_size=7, padding=3), nn.MaxPool2d(kernel_size=2, stride=2), nn.ReLU(
            inplace=True), nn.Conv2d(in_channels=8, out_channels=10, kernel_size=5, padding=2), nn.MaxPool2d(kernel_size=2, stride=2), nn.ReLU(inplace=True))

        # Regressor for the 3 * 2 affine matrix that we use
        # to make the bilinear interpolation for the spatial transformer
        self.fc_loc = nn.Sequential(nn.Linear(in_features=10 * input_height//4 * input_width//4, out_features=32,
                                              bias=True), nn.ReLU(inplace=True), nn.Linear(in_features=32, out_features=3 * 2, bias=True))

        # Initialize the weights/bias with identity transformation
        # see the article for a definition and explanation for this
        self.fc_loc[2].weight.data.fill_(0)
        self.fc_loc[2].bias.data = torch.FloatTensor([1, 0, 0, 0, 1, 0])

        # The classification network based on the transformed (cropped) image
        self.conv1 = nn.Conv2d(in_channels=input_channels,
                               out_channels=16, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(
            in_channels=16, out_channels=32, kernel_size=5, padding=2)
        self.conv2_drop = nn.Dropout2d()

        # fully connected output layers
        self.fc1_features = 32 * input_height//num_zoom//4 * input_width//num_zoom//4
        self.fc1 = nn.Linear(in_features=self.fc1_features, out_features=50)
        self.fc2 = nn.Linear(in_features=50, out_features=num_classes)

    # Spatial transformer network forward function
    def stn(self, x):
        """ Spatial Transformer Network """
        # creates distributed embeddings of the image with the location network.
        xs = self.localization(x)
        xs = xs.view(-1, 10 * self.input_height//4 * self.input_width//4)
        # project from distributed embeddings to bilinear interpolation space
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        # define the output size of the cropped tensor
        # notice that we divide the height and width with the amount of zoom
        output_size = torch.Size((x.size()[0], x.size()[1], x.size()[
                                 2]//self.num_zoom, x.size()[3]//self.num_zoom))
        # magic pytorch functions that are used for transformer networks
        # http://pytorch.org/docs/master/nn.html#torch.nn.functional.affine_grid
        grid = F.affine_grid(theta, output_size)
        # http://pytorch.org/docs/master/nn.html#torch.nn.functional.grid_sample
        x = F.grid_sample(x, grid)
        return x

    def forward(self, x):
        # transform the input
        x = self.stn(x)
        # save transformation
        l_trans1 = Variable(x.data)

        # Perform the usual forward pass
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, self.fc1_features)
        x = F.relu(self.fc1(x))
        # note use of Functional.dropout, where training must be explicitly defined (default: False)
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        # return output and batch of bilinear interpolated images
        return F.log_softmax(x, dim=1), l_trans1


net = Net()

print(type(net))
if torch.cuda.is_available():
    print('##converting network to cuda-enabled')
    net.cuda()
print(net)


# test forward pass on dummy data
x = np.random.normal(0, 1, (45, 1, 60, 60)).astype('float32')
x = Variable(torch.from_numpy(x))
if torch.cuda.is_available():
    x = x.cuda()
output = net(x)
print([x.size() for x in output])


def get_variable(x):
    """ Converts tensors to cuda, if available. """
    if torch.cuda.is_available():
        return x.cuda()
    return x


def get_numpy(x):
    """ Get numpy array for both cuda and not. """
    if torch.cuda.is_available():
        return x.cpu().data.numpy()

    return x.data.numpy()


def train_BO(net: torch.nn.Module, Input: data['X_train'],  Label: data['y_train'], parameters: Dict[str, float],) -> nn.Module:
    """
    Train the network on provided data set to find the optimzed hyperparamter settings.

    Args:
        net: initialized neural network
        Input: The image
        Label: Th label to the respective image
        parameters: dictionary containing parameters to be passed to the optimizer.
            - lr: default (0.001)
            - momentum: default (0.0)
            - weight_decay: default (0.0)
            - num_epochs: default (1)
    Returns:
        nn.Module: trained Network.
    """
    # Define the data
    X = data['X_train']
    y = data['y_train']

    # Initilize network
    net.train()

    # Define the hyperparameters

    criterion = nn.NLLLoss(reduction="sum")
    optimizer = optim.SGD(net.parameters(), lr=parameters.get("lr", 0.001), momentum=parameters.get(
        "momentum", 0.0), weight_decay=parameters.get("weight_decay", 0.0),)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=int(
        parameters.get("step_size", 30)), gamma=parameters.get("gamma", 1.0),)
    num_epochs = parameters.get("num_epochs", 1)

    # Get the number of samples and batches before starting trainging the network
    num_samples = X.shape[0]
    num_batches = int(np.ceil(num_samples / float(BATCH_SIZE)))

    for _ in range(num_epochs):
        for i in range(num_batches):
            idx = range(i*BATCH_SIZE, np.minimum((i+1)
                                                 * BATCH_SIZE, num_samples))
            X_batch_tr = get_variable(Variable(torch.from_numpy(X[idx])))
            y_batch_tr = get_variable(
                Variable(torch.from_numpy(y[idx]).long()))

            optimizer.zero_grad()
            output, _ = net(X_batch_tr)
            batch_loss = criterion(output, y_batch_tr)

            batch_loss.backward()
            optimizer.step()
            scheduler.step()
#    pass
    return net


def eval_BO(net: torch.nn.Module, Input: data['X_valid'], Label: data['y_valid'], ) -> float:
    """
    Compute classification accuracy on provided dataset to find the optimzed hyperparamter settings.

    Args:
        net: trained neural network
        Input: The image
        Label: Th label to the respective image
    Returns:
        float: classification accuracy
    """
    # Define the data
    X = data['X_valid']
    y = data['y_valid']

    # Pre-locating memory
    pred_list = []

    # Get the number of samples and batches before testing the network
    num_samples = X.shape[0]
    num_batches = int(np.ceil(num_samples / float(BATCH_SIZE)))
    net.eval()

    with torch.no_grad():
        for i in range(num_batches):

            idx = range(i*BATCH_SIZE, np.minimum((i+1)
                                                 * BATCH_SIZE, num_samples))
            X_batch_val = get_variable(Variable(torch.from_numpy(X[idx])))
            output, transformation = net(X_batch_val)
            pred_list.append(get_numpy(output))

    # Calculating the accuracy
    preds = np.concatenate(pred_list, axis=0)
    preds = np.argmax(preds, axis=-1)
    acc = np.mean(preds == y)

#    pass
    return acc



def evaluate_Hyperparameters(parameterization):
    """
    Train and evaluate the network to find the best parameters 
    Args:
        parameterization: The hyperparameters that should be evaluated
    Returns:
        float: classification accuracy
    """
    net = Net()
    net = train_BO(net=net, Input=data['X_train'],
                   Label=data['y_train'], parameters=parameterization,)
#    pass
    return eval_BO(net=net, Input=data['X_valid'], Label=data['y_valid'],)


def evaluate_Hyperparameters_II(parameterization):
    """
    Train and evaluate the network to find the best parameters 
    Args:
        parameterization: The hyperparameters that should be evaluated
    Returns:
        float: classification accuracy
    """
    net = NET
    net = train_BO(net=net, Input=data['X_train'],
                   Label=data['y_train'], parameters=parameterization,)
#    pass
    return eval_BO(net=net, Input=data['X_valid'], Label=data['y_valid'],)



#best_parameters, values, experiment, model = optimize(parameters=[{"name": "lr", "type": "range", "bounds": [1e-5, 0.1], "log_scale": False}, {"name": "momentum", "type": "range", "bounds": [0.2, 0.6]}, ], evaluation_function=evaluate_Hyperparameters, objective_name='accuracy',)


# Saving the results from the optimization
#BE = []
#ME = []
#CO = []

#means, covariances = values

#BE.append(best_parameters)
#ME.append(means)
#CO.append(covariances)

# Printing the results of the hyperparamter optimization
#print(best_parameters)
#print(means, covariances)

# Findin the best hyperparameter for training the network
#data1 = experiment.fetch_data()
#df = data1.df
#best_arm_name = df.arm_name[df['mean'] == df['mean'].max()].values[0]
#best_arm = experiment.arms_by_name[best_arm_name]
#best_arm

class best_arm1:
    parameters = {}
    parameters["lr"] = 0.001
    parameters["momentum"] = 0.001
    parameters["weight_decay"] = 0.001

best_arm = best_arm1()




#render(plot_contour(model=model, param_x='lr', param_y='momentum', metric_name='accuracy'))


#best_objectives = np.array([[trial.objective_mean*100 for trial in experiment.trials.values()]])
#best_objective_plot = optimization_trace_single_method(y=np.maximum.accumulate( best_objectives, axis=1), title="Model performance vs. # of iterations", ylabel="Classification Accuracy, %",)

#render(best_objective_plot)


def train_epoch(net: torch.nn.Module, Input: data['X_valid'], Label: data['y_valid'], parameters: Dict[str, float],) -> float:
    """
    Train the network with the optimized hyperparamters.

    Args:
        Input: The image
        Label: The label to the respective image
        Param: Parameters the best paramters for training the network
    Returns:
        float: the mean cost and the classification accuracy
    """
    # Define the data
    X = data['X_valid']
    y = data['y_valid']

    # Initilize network
    net.train()

  # Define the hyperparameters
    optimizer = optim.SGD(net.parameters(), lr=parameters.get("lr", 0.001), momentum=parameters.get(
        "momentum", 0.0), weight_decay=parameters.get("weight_decay", 0.0),)

    # Pre-locating memory
    costs = []
    correct = 0

    # Get the number of samples and batches before testing the network
    num_samples = X.shape[0]
    num_batches = int(np.ceil(num_samples / float(BATCH_SIZE)))

    for i in range(num_batches):
        idx = range(i*BATCH_SIZE, np.minimum((i+1)*BATCH_SIZE, num_samples))
        X_batch_tr = get_variable(Variable(torch.from_numpy(X[idx])))
        y_batch_tr = get_variable(Variable(torch.from_numpy(y[idx]).long()))

        optimizer.zero_grad()
        output, _ = net(X_batch_tr)
        batch_loss = criterion(output, y_batch_tr)

        batch_loss.backward()
        optimizer.step()

        costs.append(get_numpy(batch_loss))
        preds = np.argmax(get_numpy(output), axis=-1)
        correct += np.sum(get_numpy(y_batch_tr) == preds)
    #   pass
    return np.mean(costs), correct / float(num_samples)


def eval_epoch(net: torch.nn.Module, Input: data['X_valid'],  Label: data['y_valid'],) -> float:
    """
    Compute classification accuracy on provided dataset to the optimized network.

    Args:
        Input: The image
        Label: Th label to the respective image
    Returns:
        float: classification accuracy
    """
    # Define the data
    X = data['X_valid']
    y = data['y_valid']

    # Pre-locating memory
    pred_list = []

    # Get the number of samples and batches before testing the network
    num_samples = X.shape[0]
    num_batches = int(np.ceil(num_samples / float(BATCH_SIZE)))
    net.eval()

    for i in range(num_batches):
        idx = range(i*BATCH_SIZE, np.minimum((i+1)*BATCH_SIZE, num_samples))
        X_batch_val = get_variable(Variable(torch.from_numpy(X[idx])))
        output, _ = net(X_batch_val)
        pred_list.append(get_numpy(output))

    # Calculating the accuracy
    preds = np.concatenate(pred_list, axis=0)
    preds = np.argmax(preds, axis=-1)
    acc = np.mean(preds == y)

#    pass
    return acc



# Using the old network again with the new hyperparamter-setting
net = Net()

# Saving the results from the optimization
valid_accs, train_accs, test_accs = [], [], []
criterion = nn.NLLLoss(reduction="sum")

n = 0
while n < NUM_EPOCHS:
    n += 1
    try:
        print("Epoch %d:" % n)

        train_cost, train_acc = train_epoch( net=net, Input=data['X_train'],Label=data['y_train'], parameters=best_arm.parameters,)
        valid_acc = eval_epoch( net=net, Input=data['X_valid'], Label=data['y_valid'],)
        test_acc = eval_epoch(net=net, Input=data['X_test'], Label=data['y_test'],)
        valid_accs += [valid_acc]
        test_accs += [test_acc]
        train_accs += [train_acc]

        print("train cost {0:.2}, train acc {1:.2}, val acc {2:.2}, test acc {3:.2}".format(train_cost, train_acc, valid_acc, test_acc))

        """ For every N new hyperparameters will be calculated for training the network"""
        if n % 5 == 0:
            # Saving the network trained network so far, so it can be used when the optimization is done.
            NET = net
            NET_II = net

            # Findinf the best hypperparameter again.
            best_parameters, values, experiment, model = optimize(parameters=[{"name": "lr", "type": "range", "bounds": [1*10**(-5), 0.1], "log_scale": False}, {"name": "momentum", "type": "range", "bounds": [0.2, 0.6]}, ], evaluation_function=evaluate_Hyperparameters_II, objective_name='accuracy',)

            # Getting the best hyperparameters
            data1 = experiment.fetch_data()
            df = data1.df
            best_arm_name = df.arm_name[df['mean'] == df['mean'].max()].values[0]
            best_arm = experiment.arms_by_name[best_arm_name]

            # Saving the results from the optimization
            BE.append(best_parameters)
            ME.append(means)
            CO.append(CO)

            # Using the old network again with the new hyperparamter-setting
            net = NET_II

    except KeyboardInterrupt:
        print('\nKeyboardInterrupt')
        break

