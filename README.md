# CMSE802_spring2020_hyperparamterop

# Overview #

In the last couple of years, there has been a major shift in training and using
convolutional neural networks as a second opinion in medical detection and
diagnostic. It has been proven in some medical cases that the neural networks 
outperform the detectors. Thus, they are thought to be able to work as a second 
opinion to minimize the time used on detecting and diagnosing a patient, while
increasing sensitivity and specificity. Convolution neural networks can be
trained by fitting the network of weights iteratively to a wished outcome by a 
known input. For each convolution neural networks, a few hyperparameters have 
to be picked. These parameters are often chosen before training and they are 
not changed under training. These parameters are picked based on experience and
retraining the model a couple of times. This project will look into finding a 
method, either grid search or adaptive selection, to choose the hyperparameters 
while training the model. Thus, retraining the model is no longer necessary. 

The optimization algorithm will be used in a future convolution neural network 
trained on medical data.  

# Program Description #

In this course, a "dummy" convolution neural network will be used together with 
the MNIST data set for training. The convolutional neural network is written by 
Lars Maaloqe, Soqren Kaae Soenderby, Casper Soenderby and updated by Toke Faurby 
to Pytorch and given on Github for the course 04256 Deep Learning at Danish 
Technical University. It can be used for free. The MNIST data set is also 
public and it consists of pictures of handwritten numbers and the belonging 
label. The goal of this project is to make an optimization algorithm, which will 
update hyperparameters to get the best outcome when the model is training. For 
visualization of how the optimization algorithm picks the hyperparameters, a 
heatmap (see the above picture) for specific epochs will be made. The heatmap 
will have a range of values for the axes and the heatmap will describe how 
accurate the model becomes for the chosen hyperparameters. A graph of the 
chosen values of hyperparameters for each epoch will also be made. For proof 
of concept, the network will be run e.g. 10 times with and without the 
optimization algorithm. The accuracy will be saved and inspected visually for 
significant differences between the two methods.
