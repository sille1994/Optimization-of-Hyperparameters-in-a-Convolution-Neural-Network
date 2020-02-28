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

Webside: https://github.com/DeepLearningDTU/02456-deep-learning

# Bayesian Optimization #

Bayesian Optimization (BO) is a methods to maximize the evaluating of "black 
box" functions, this could be Machine Learning or Deep Learning algorithms, 
without having acces to these algorithms. Thus, BO takes the algorithm and test 
it at a sequence of test points to determine the optimal values for e.g. 
hyperparamters.

This is done by approximations, since we cannot solve the problem analyticly. 
The approximation is a surregate model (a probabilistic model) based om the 
results with the associated hyperparameter. Here, a Gaussian model is fitted to 
the outcome. This will give us a mean and vairance. An acquisition function is 
then used to look at different trade-offs of picking known maximas and explore 
uncertain locations in the hyperparamter space. These steps are iterated, thus 
we should get a better and better approximation until the maximum number of 
ietrations is met. The best results can then be found with the respictive 
hyperparameter(s).

References
Citing: https://mlconf.com/blog/lets-talk-bayesian-optimization/


# Running the programe #

Install Linux: 
conda install pytorch torchvision cpuonly -c pytorch
conda install numpy==1.16.1
conda install matplotlib==3.1.0
conda install botorch -c pytorch -c gpytorch
pip3/pip install ax-platform

Install Mac:
conda install pytorch torchvision -c pytorch
conda install numpy==1.16.1
conda install matplotlib==3.1.0
conda install botorch -c pytorch -c gpytorch
pip3/pip install ax-platform

Install Windows:
conda install pytorch torchvision cpuonly -c pytorch
conda install numpy==1.16.1
conda install matplotlib==3.1.0
conda install botorch -c pytorch -c gpytorch
pip3/pip install ax-platform

Torch and Torchvison should be respectively the version  1.3.1
and 0.4.2.


To run the unit test;
Open the terminal and make sure you are in the 
"Hyperparameter Optimization"- folder. 
Then run: "python test_Project_CSME802.py"



