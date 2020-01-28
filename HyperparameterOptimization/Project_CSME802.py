#!/usr/bin/env python
# coding: utf-8

# 1. Loading 

# In[3]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import numpy as np

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.autograd import Variable

from ray import tune
from ray.tune.examples.mnist_pytorch import get_data_loaders, ConvNet, train, test


# In[ ]:


get_ipython().system('if [ ! -f mnist_cluttered_60x60_6distortions.npz ]; then wget -N https://www.dropbox.com/s/rvvo1vtjjrryr7e/mnist_cluttered_60x60_6distortions.npz; else echo "mnist_cluttered_60x60_6distortions.npz already downloaded"; fi')


# 2. Load data and minding to be smaller. 

# In[3]:


NUM_EPOCHS = 50
BATCH_SIZE = 256
DIM = 60
NUM_CLASSES = 10
mnist_cluttered = "mnist_cluttered_60x60_6distortions.npz"


# In[4]:


def load_data():
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

    return dict(
        X_train=np.asarray(X_train, dtype='float32'),
        y_train=y_train.astype('int32'),
        X_valid=np.asarray(X_valid, dtype='float32'),
        y_valid=y_valid.astype('int32'),
        X_test=np.asarray(X_test, dtype='float32'),
        y_test=y_test.astype('int32'),
        num_examples_train=X_train.shape[0],
        num_examples_valid=X_valid.shape[0],
        num_examples_test=X_test.shape[0],
        input_height=X_train.shape[2],
        input_width=X_train.shape[3],
        output_dim=10,)
data = load_data()

idx = 0
canvas = np.zeros((DIM*NUM_CLASSES, NUM_CLASSES*DIM))
for i in range(NUM_CLASSES):
    for j in range(NUM_CLASSES):
        canvas[i*DIM:(i+1)*DIM, j*DIM:(j+1)*DIM] = data['X_train'][idx].reshape((DIM, DIM))
        idx += 1
plt.figure(figsize=(10, 10))
plt.imshow(canvas, cmap='gray')
plt.title('Cluttered handwritten digits')
plt.axis('off')

plt.show()


# 3. Defining the Convolutional Neural Network

# In[ ]:





# In[6]:


class Net(nn.Module):
    
    def __init__(self, input_channels, input_height, input_width, num_classes, num_zoom=3):
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
        self.localization = nn.Sequential(
            nn.Conv2d(in_channels=input_channels,
                      out_channels=8, 
                      kernel_size=7, 
                      padding=3),
            nn.MaxPool2d(kernel_size=2, 
                         stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=8, 
                      out_channels=10, 
                      kernel_size=5, 
                      padding=2),
            nn.MaxPool2d(kernel_size=2,
                         stride=2),
            nn.ReLU(inplace=True)
        )

        # Regressor for the 3 * 2 affine matrix that we use 
        # to make the bilinear interpolation for the spatial transformer
        self.fc_loc = nn.Sequential(
            nn.Linear(in_features=10 * input_height//4 * input_width//4, 
                      out_features=32,
                      bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=32, 
                      out_features=3 * 2,
                      bias=True)
        )

        # Initialize the weights/bias with identity transformation
        # see the article for a definition and explanation for this
        self.fc_loc[2].weight.data.fill_(0)
        self.fc_loc[2].bias.data = torch.FloatTensor([1, 0, 0, 0, 1, 0])
        
        # The classification network based on the transformed (cropped) image
        self.conv1 = nn.Conv2d(in_channels=input_channels,
                               out_channels=16,
                               kernel_size=5,
                               padding=2)
        self.conv2 = nn.Conv2d(in_channels=16, 
                               out_channels=32,
                               kernel_size=5,
                               padding=2)
        
        self.conv2_drop = nn.Dropout2d()
        
        # fully connected output layers
        self.fc1_features = 32 * input_height//num_zoom//4 * input_width//num_zoom//4
        self.fc1 = nn.Linear(in_features=self.fc1_features, 
                             out_features=50)
        self.fc2 = nn.Linear(in_features=50,
                             out_features=num_classes)

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
        output_size = torch.Size((x.size()[0],
                                  x.size()[1], 
                                  x.size()[2]//self.num_zoom,
                                  x.size()[3]//self.num_zoom))
        # magic pytorch functions that are used for transformer networks
        grid = F.affine_grid(theta, output_size) # http://pytorch.org/docs/master/nn.html#torch.nn.functional.affine_grid
        x = F.grid_sample(x, grid) # http://pytorch.org/docs/master/nn.html#torch.nn.functional.grid_sample
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


net = Net(1, DIM, DIM, NUM_CLASSES)
if torch.cuda.is_available():
    print('##converting network to cuda-enabled')
    net.cuda()
print(net)


# In[7]:


# test forward pass on dummy data
x = np.random.normal(0,1, (45, 1, 60, 60)).astype('float32')
x = Variable(torch.from_numpy(x))
if torch.cuda.is_available():
    x = x.cuda()
output = net(x)
print([x.size() for x in output])


# In[8]:





# In[9]:


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


# In[10]:




def train_epoch(X, y,):
    num_samples = X.shape[0]
    num_batches = int(np.ceil(num_samples / float(BATCH_SIZE)))
    costs = []
    correct = 0
    net.train()
    for i in range(num_batches):
        if i % 10 == 0:
            print("{}, ".format(i), end='')
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
    print()
    return np.mean(costs), correct / float(num_samples)

def eval_epoch(X, y):
    num_samples = X.shape[0]
    num_batches = int(np.ceil(num_samples / float(BATCH_SIZE)))
    pred_list = []
    transform_list = []
    net.eval()
    for i in range(num_batches):
        if i % 10 == 0:
            print("{}, ".format(i), end='')
        idx = range(i*BATCH_SIZE, np.minimum((i+1)*BATCH_SIZE, num_samples))
        X_batch_val = get_variable(Variable(torch.from_numpy(X[idx])))
        output, transformation = net(X_batch_val)
        pred_list.append(get_numpy(output))
        transform_list.append(get_numpy(transformation))
    transform_eval = np.concatenate(transform_list, axis=0)
    preds = np.concatenate(pred_list, axis=0)
    preds = np.argmax(preds, axis=-1)
    acc = np.mean(preds == y)
    print()
    return acc, transform_eval


# In[11]:


n = 0


# In[ ]:


valid_accs, train_accs, test_accs = [], [], []

analysis = tune.run(
    train_epoch, config={"lr": tune.grid_search([0.001, 0.01, 0.1])})

while n < NUM_EPOCHS:
    n += 1
    try:
        print("Epoch %d:" % n)
        print('train: ')
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(net.parameters(),lr=config["lr"])
        train_cost, train_acc = train_epoch(data['X_train'], data['y_train'])
        tune.track.log(mean_accuracy=train_acc)

        print('valid ')
        valid_acc, valid_trainsform = eval_epoch(data['X_valid'], data['y_valid'])
        print('test ')
        test_acc, test_transform = eval_epoch(data['X_test'], data['y_test'])
        valid_accs += [valid_acc]
        test_accs += [test_acc]
        train_accs += [train_acc]

        print("train cost {0:.2}, train acc {1:.2}, val acc {2:.2}, test acc {3:.2}".format(
                train_cost, train_acc, valid_acc, test_acc))
    except KeyboardInterrupt:
        print('\nKeyboardInterrupt')
        break


# In[ ]:


print("Best config: ", analysis.get_best_config(metric="mean_accuracy"))


# In[ ]:


plt.figure(figsize=(9, 9))
plt.plot(1 - np.array(train_accs), label='Training Error')
plt.plot(1 - np.array(valid_accs), label='Validation Error')
plt.plot(1 - np.array(test_accs), label='Test Error')

plt.legend(fontsize=20)
plt.xlabel('Epoch', fontsize=20)
plt.ylabel('Error', fontsize=20)
plt.show()


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
from pylab import *
import random

images, labels = data['X_test'], data['y_test']
num_samples = images.shape[0]
num_batches = int(np.ceil(num_samples / float(BATCH_SIZE)))
pred_list = []
transform_list = []
net.eval()
for i in range(1):
        idx = range(i*BATCH_SIZE, np.minimum((i+1)*BATCH_SIZE, num_samples))
        X_batch_val = get_variable(Variable(torch.from_numpy(images[idx])))
        output, transformation = net(X_batch_val)
        pred_list.append(get_numpy(output))

labels = labels[idx]  
preds = np.concatenate(pred_list, axis=0)
preds = np.argmax(preds, axis=-1)
predicted = preds

for i in range(5):
        idx_n = random.randint(0, len(images[idx]))
        print('GroundTruth:', ' '.join('%5s' % labels[idx_n]), ',  Predicted:', ' '.join('%5s' % predicted[idx_n]))
        image= X_batch_val[idx_n]
        image = image[0,:,:]
        imshow(image)
        plt.show()

                        
    

