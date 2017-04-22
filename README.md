# CarND-Traffic-Sign-Classifier-Project
My submission for the Udacity Self-Driving Car Nanodegree program Project 2 - Traffic Sign Classifier

## Dataset Exploration
### Dataset Summary
Number of training examples = 34799
Number of validation examples = 4410
Number of testing examples = 12630
Image data shape = (32, 32, 1)
Number of classes = 43

### Exploratory Visualization
A few random images from the dataset were displayed and a histogram of distrubtion of training data over all classes was printed to give a gist of dataset.

## Design and Test a Model Architecture
### Preprocessing

My dataset preprocessing consisted of:

1. Converting to grayscale - This helps in reduction of training time.
2. Normalizing the data to the range (-1,1) - If we don't normalize the data, a wider distribution in the data would make it more difficult to train using a singlar learning rate. Different features could encompass far different ranges and a single learning rate might make some weights diverge from optimum.
3. Data augmentation -  From the histogram, we could see the training data was distrubted very un-evenly across classes. It is widely accepted that a model is trained better when all classes have sufficient number of training examples, hence more training examples were created for the sparse classes by using 5 different augmentation techniques over existing examples. These techniques were random_translate, random_scaling,random_warp, random_brightness.


### Model Architecture
I started with a base architecture of LeNet and then modified it into a new function LeNet2 which has the following layers:
1. 5x5 convolution (32x32x1 in, 28x28x6 out)
2. ReLU
3. 2x2 max pool (28x28x6 in, 14x14x6 out)
4. 5x5 convolution (14x14x6 in, 10x10x16 out)
5. ReLU
6. 2x2 max pool (10x10x16 in, 5x5x16 out)
7. 5x5 convolution (5x5x6 in, 1x1x400 out)
8. ReLu
9. Flatten layers from numbers 8 (1x1x400 -> 400) and 6 (5x5x16 -> 400)
Concatenate flattened layers to a single size-800 layer
Dropout layer
Fully connected layer (800 in, 43 out)

### Model Training
I used the Adam optimizer. The settings used were:
batch size: 100
epochs: 60
learning rate: 0.0009
mu: 0
sigma: 0.1
dropout keep probability: 0.5

### Solution Approach
I Started with LeNet architecture and through several trials, first finalized the optimizer's hyperparameters.
After that, I played around with different layers to get a better validation accuracy.
The Test Accuracy was 94.7%


## Test a Model on New Images

### Acquiring New Images
I acquired 6 new images of german signs from internet

### Performance on New Images
The model predicted with an accuracy of 100% on new images. 
Although this doesn't mean that it would work well for all images.

### Model Certainty - Softmax Probabilities
The softmax probabilities were 100% correct for 5 out of 6 new images. For 1 image, softmax probability was 99% in favour of the correct class, which is still pretty remarkable.
