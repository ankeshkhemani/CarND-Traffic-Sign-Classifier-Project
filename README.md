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
As described in the lessons, the LeNet architecture is very well suited for image recognition and is a good starting point for traffic sign classification.
I started with a standard architecture of LeNet which has the following layers:

1.  Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x6.
2.          Activation.
3.          Pooling. Input = 28x28x6. Output = 14x14x6.
4.  Layer 2: Convolutional.Input = (Layer 1 output)14x14x6 Output = 10x10x16.
5.         Activation.
6.          Pooling. Input = 10x10x16. Output = 5x5x16.        
7.  Layer 3: Convolutional.Input = (Layer 2 output)5x5x16 Output = 1x1x400.
8.          Activation.
9.  L2_Flatten: Input = (Layer 2 output)5x5x16  Output = 400.
10. L3_Flatten: Input = (Layer 3 output)1x1x400. Output = 400.
11. Concat L2_Flatten and L3_Flatten: Input = 400 + 400. Output = 800
12. Dropout
13. Layer 4: Fully Connected. Input = 800. Output = 43 (Logits)



### Model Training
I used the Adam optimizer. The settings used were:
batch size: 500
epochs: 50
learning rate: 0.005
mu: 0
sigma: 0.1
dropout keep probability: 0.5

### Solution Approach

Initially, I trained the network with epoch=15 and batch size=50 and learning rate = 0.01
It was noticed that the validation accuracy was fluctuating and reached only 0.84, which points that it may be diverging adhoc due to a higher learning rate.

So I trained the network with epoch=15, batch size=50 and learning rate = 0.001 and attained a validation accuracy of 0.94 but the test accuracy was 0.92. This is probably acceptable for submission but the difference between validation and test accuracy looked suspicious.

So I trained the network with epoch=15, a higher batch size of 500 and faster learning rate =0.01 and attained a validation accuracy of 0.949 and a test accuracy of 0.932. Still some difference.

So finally I trained the network with higher epoch of 50, batch size of 500 and medium learning rate 0.005 and attained a validation accuracy of 0.972 and a test accuracy of 0.948. This is the final solution I am submitting.


## Test a Model on New Images

### Acquiring New Images
I acquired 6 new images of german signs from internet. They can be found in the project sub-directory "new-signs-data"
https://github.com/ankeshkhemani/CarND-Traffic-Sign-Classifier-Project/blob/master/new-traffic-signs/1x.png
1x.png was taken from google search, has a bright background and looks similar to other signs.

https://github.com/ankeshkhemani/CarND-Traffic-Sign-Classifier-Project/blob/master/new-traffic-signs/2x.png
2x.png is quite hazy as the photo was taken from a distance using google street view and then manually resized to 32x32

https://github.com/ankeshkhemani/CarND-Traffic-Sign-Classifier-Project/blob/master/new-traffic-signs/3x.png
3x.png again is from google street view and manually resized and has complex figure similar to other signs.

https://github.com/ankeshkhemani/CarND-Traffic-Sign-Classifier-Project/blob/master/new-traffic-signs/4x.png
4x.png was taken from google search, has a forest in background adding to noise.

https://github.com/ankeshkhemani/CarND-Traffic-Sign-Classifier-Project/blob/master/new-traffic-signs/5x.png
5x.png has varying brightness and has a complex figure similar to other signs.

https://github.com/ankeshkhemani/CarND-Traffic-Sign-Classifier-Project/blob/master/new-traffic-signs/6x.png
6x.png is a simple one taken from google search.


### Performance on New Images
The model predicted with an accuracy of 50% on new images.
The model performed very poorly on images taken from google street view, in my opinion primarily because they were manually scaled with a lot of warp and were hazy too.
It also labelled a clearly visible road work sign wrongly, this class has a complex figure which is similar to other signs.

Image  -           Actual                                Prediction

1x.png - Right-of-way at the next intersection    Right-of-way at the next intersection 

2x.png - Go straight or left                      Speed limit (60km/h)

3x.png - Road work                                Wild animals crossing

4x.png - General caution                          General caution

5x.png - Road work                                Bicycles crossing

6x.png - Speed limit (60km/h)                     Speed limit (60km/h)


### Model Certainty - Softmax Probabilities
The softmax probabilities give an insight as to what went wrong with some images.
Here are the results of the prediction:

Image  -           Actual                                Prediction

1x.png - Right-of-way at the next intersection    (100%)Right-of-way at the next intersection 

2x.png - Go straight or left                      (96%)Speed limit (60km/h), (2%)Speed limit (80km/h), (2%)Priority road

3x.png - Road work                                (100%)Wild animals crossing

4x.png - General caution                          (100%)General caution

5x.png - Road work                                (49%)Bicycles crossing, (47%)Road work, (4%)Bumpy road

6x.png - Speed limit (60km/h)                     (100%)Speed limit (60km/h)


We observe that all correctly labeled classes have 100% probability. The ones that were manually scaled do especially bad, even giving 100% softmax probability to the wrong class. One road work sign which is clearly visible to eyes still gives 49% probability to Bicycles crossing and 47% to road work. This means the model has not trained well on the minute features of roadwork.

# References: 
For data augmentation and some other code, reference has been taken from Jeremy's project code at https://github.com/jeremy-shannon/CarND-Traffic-Sign-Classifier-Project
