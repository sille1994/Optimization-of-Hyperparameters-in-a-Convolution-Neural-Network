# CMSE802_spring2020_hyperparamterop

# Overview #

In the last couple of years, there has been a major shift in training and using
convolutional neural networks as a second opinion in medical detection and
diagnostic. It has been proven in some medical cases that the neural networks 
outperform the detectors [1–7]. Thus, they are thought to be able to work as a second 
opinion to minimize the time used on detecting and diagnosing a patient, while
increasing sensitivity and specificity. Convolution neural networks can be
trained by fitting the network of weights iteratively to a wished outcome by a 
known input [8]. For each convolution neural networks, a few hyperparameters have 
to be picked [8]. These parameters are often chosen before training and they are 
not changed under training. These parameters are picked based on experience and
retraining the model a couple of times. This project will look into finding the bedst 
hyperparameters using Bayersian optimization. Thus, retraining the model is no longer necessary. 

The optimization algorithm will be used in a future convolution neural network 
trained on medical data.  

# Program Description #

In this course, a "dummy" convolution neural network will be used together with 
the MNIST data set for training. The convolutional neural network is written by 
Lars Maaloqe, Soqren Kaae Soenderby, Casper Soenderby and updated by Toke Faurby 
to Pytorch and given on Github for the course 04256 Deep Learning at Danish 
Technical University. It can be used for free. The MNIST data set is also 
public and it consists of pictures of handwritten numbers and the belonging 
label. The reasoning behind using the the "dummy" convolution neural network 
is that most projects involicing integrating neural networks or convolution neural 
networs are designed specific for the predicting diseases from biomarkes, signals, 
or medical images. Thus, the networks, and the respective functions used to train the 
network are often made by the scientific research group by themself. 
The goal of this project is to make an optimization algorithm, which will 
update hyperparameters to get the best outcome when the network is training. For proof 
of concept, the network will be run e.g. 10 times with and without the 
optimization algorithm. The accuracy will be saved and inspected visually for 
significant differences between the two methods.

"Dummy" convolution neural networt: <br/>
Original Webside: https://github.com/DeepLearningDTU/02456-deep-learning<br/>
Updated to Pytorch: https://github.com/DeepLearningDTU/02456-deep-learning-with-PyTorch

# Bayesian Optimization #

Bayesian Optimization (BO) is a methods to maximize the evaluating of "black 
box" functions, this could be Machine Learning or Deep Learning algorithms, 
without having acces to these algorithms. Thus, BO takes the algorithm and test 
it at a sequence of test points to determine the optimal values for e.g. 
hyperparamters [9].

This is done by approximations, since we cannot solve the problem analyticly. 
The approximation is a surregate model (a probabilistic model) based om the 
results with the associated hyperparameter. Here, a Gaussian model is fitted to 
the outcome. This will give us a mean and vairance. An acquisition function is 
then used to look at different trade-offs of picking known maximas and explore 
uncertain locations in the hyperparamter space. These steps are iterated, thus 
we should get a better and better approximation until the maximum number of 
ietrations is met [9] . The best results can then be found with the respictive 
hyperparameter(s).


# Running the programe #

Installing: 
Use one of the following ways to install the packages used in Python. Python version 3.7 is used in this project.


Open the terminal and rhen run:<br/>
    - conda install pytorch torchvision cpuonly -c pytorch<br/>
    - conda install numpy==1.16.1<br/>
    - conda install matplotlib==3.1.0<br/>
    - conda install botorch -c pytorch -c gpytorch<br/>
    - pip install ax-platform<br/>

Or 

Open the terminal and make sure you are in the 
"cmse802_spring2020_hyperparamterop"- folder. <br/>
Then run:  <br/>
    - conda env create --prefix ./envs --file ./Software/requirements.yml<br/>
    - conda activate ./envs<br/>

Or <br/>

Open the terminal and make sure you are in the 
"cmse802_spring2020_hyperparamterop"- folder. <br/>
Then run:  <br/>
    - make init  <br/>
    - conda activate ./envs  <br/>


# #

To run the unit test;
Open the terminal and make sure you are in the 
"cmse802_spring2020_hyperparamterop"- folder. <br/>
Then run: make test

# #
To run pylint:<br/>
Pylint has troubles with Torch. There is no
[solution](https://github.com/pytorch/pytorch/issues/701), thus
--disable=no-member is added.<br/>
Open the terminal and make sure you are in the 
"cmse802_spring2020_hyperparamterop"- folder. <br/>
Then run: make lint

##

To make the documentation:<br/>
Open the terminal and make sure you are in the 
"cmse802_spring2020_hyperparamterop"- folder. <br/>
Then run: make doc

##

To run the code:<br/>
Open the terminal and make sure you are in the 
"cmse802_spring2020_hyperparamterop"- folder. <br/>
Then run: make runcode

# #

The video presentation can be seen at: https://youtu.be/WCORT4HQZpU

# References #

[1] Sindhu Ramachandran S, Jose George, and Shibon Skaria. “Using YOLO baseddeep learning network for real time detection and localization of lung nodulesfrom low dose CT scans”. In: February 2018 (2019).doi:10.1117/12.2293699.[14]Aiden Nibali, Zhen He, and Dennis Wollersheim. “Pulmonary nodule classifica-tion with deep residual networks”. eng. In:International Journal of ComputerAssisted Radiology and Surgery12.10 (2017), pages 1799–1808.issn: 1861-6410.<br/>
[2] Wentao Zhu et al. “DeepLung: Deep 3D dual path nets for automated pul-monary nodule detection and classification”. In:Proceedings - 2018 IEEE Win-ter Conference on Applications of Computer Vision, WACV 20182018-Janua(2018), pages 673–681.doi:10.1109/WACV.2018.00079.<br/>
[3] Manu Sharma, Jignesh S Bhatt, and Manjunath V Joshi. “Early detection oflung cancer from classification using deep learning”. In: April 2018 (2019).doi:10.1117/12.2309530.<br/>
[4] Emre Dandil et al. “Artificial neural network-based classification system forlung nodules on computed tomography scans”. eng. In:2014 6th InternationalConference of Soft Computing and Pattern Recognition (SoCPaR). IEEE, 2014,pages 382–386.isbn: 9781479959341.<br/>
[5] Jinsa Kuruvilla and K Gunavathi. “Lung cancer classification using neural net-works for CT images.” eng. In:Computer methods and programs in biomedicine113.1 (2014), pages 202–209.issn: 1872-7565.url:http://search.proquest.com/docview/1461341321/.<br/>
[6] Carmen Krewer et al. “Immediate effectiveness of single-session therapeutic in-terventions in pusher behaviour.” eng. In:Gait posture37.2 (2013), pages 246–250.issn: 1879-2219.url:http://search.proquest.com/docview/1282049046/.<br/>
[7] L.B. Nascimento, A.C. De Paiva, and A.C. Silva. “Lung nodules classificationin CT images using Shannon and Simpson Diversity Indices and SVM”. In:volume 7376. 2012, pages 454–466.isbn: 9783642315367.<br/>
[8] Hargrave, Marschall. 2019. “Deep Learning.” April 30. https://www.investopedia.com/terms/d/deep-learning.asp.<br/>
[9] Aravikumar, Meghan. 2018. "Let’s Talk Bayesian Optimization." November 16. https://mlconf.com/blog/lets-talk-bayesian-optimization/.

# Note #

There are more python files made for the code. They are respectively used for the main code, unit test, linting, and documentation. In testing, a separate python file is made, so Bayesian optimization and training are not done. This would take half an hour if a separate file the main functions were not made. The auto-documentation also has to be used with a separate python file, since it cannot handle the ax module. Thus, a separate file is also made.
