#**Traffic Sign Recognition** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/visualization.png "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image4]: ./examples/test_01.jpg "Traffic Sign 1"
[image5]: ./examples/test_02.jpg "Traffic Sign 2"
[image6]: ./examples/test_03.jpg "Traffic Sign 3"
[image7]: ./examples/test_04.jpg "Traffic Sign 4"
[image8]: ./examples/test_05.jpg "Traffic Sign 5"
[image9]: ./examples/probs.png "Traffic Sign probs"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set and identify where in your code the summary was done. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

The code for this step is contained in the second code cell of the IPython notebook.  

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset and identify where the code is in your code file.

The code for this step is contained in the forth code cell of the IPython notebook.  

Here is an exploratory visualization of the data set. It is a bar chart showing how the data ...

![alt text][image1]

### Design and Test a Model Architecture

####1. Describe how, and identify where in your code, you preprocessed the image data. What tecniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.

The code for this step is contained in the fifth code cell of the IPython notebook.
I use TF pipeline to preprocess images

As a first step, I decided to convert the images to hsv just because rgb sucks. But I still hope to extract some data from color channel of the image.

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image2]

As a last step, I normalized the image data because NN works that way. NNs require normalization, in other way learning rate would be much slower.

#### 2. Describe how, and identify where in your code, you set up training, validation and testing data. How much data was in each set? Explain what techniques were used to split the data into these sets. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, identify where in your code, and provide example images of the additional data)

For some reason I already had validation file in the archive I've dowloaded. So all I've done is just batching and shuffling, because without it the algorithm tends to overfit. 

I don't augment the dataset because let's face it - it's a dirty hack and we need to invent better NN architectures. that would support rotation, scale, tansition equvarience.


#### 3. Describe, and identify where in your code, what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The code for my final model is located in the fifth cell of the ipython notebook. 
in the end of the day my model is just a cut off from Pierre Sermanet and Yann LeCun model (http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf),

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 HSV image   							| 
| Convolution 5x5     	| 1x1 stride, same padding, outputs 28x28x32 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x32 				|
| Convolution 5x5     	| 1x1 stride, same padding, outputs 10x10x64 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x64   				|
| Fully connected		| (5x5x64) > 160								|
| Fully connected		| 160 > 43										|
| Softmax				| to predict label								|
|						|												|
 


#### 4. Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The code for training the model is located in the eigth cell of the ipython notebook. 

To train the model, I used an small batches, with much larger batches model tends to overfit. 
Used default AdamOptimizer. Current model goes to 93% accuracy in 6 epochs, and it seems more epochs don't change this number dramatically.
I've tried to run over 100 epochs and it seems the model trends to overfit on large amount of learning epochs.

#### 5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The code for calculating the accuracy of the model is located in the ninth cell of the Ipython notebook.

My final model results were:
* training set accuracy of 0.98-099
* validation set accuracy of 0.93-0.95
* test set accuracy of 0.93

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?

LeNet5

* What were some problems with the initial architecture?

most likely too few neurons on first layers. and too many full conv layers

* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to over fitting or under fitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.

adding another conv layer didn't improve accuracy, even dropped it to 90% That is strange and unfortunately is hard to find answer why it happened.
d
ropout layer is required in case of overfit, but I belive it is better to reduce number of neurons (number of computations), so model have no ability to remember test dataset.

I've left relu because it usually used with conv layers. Don't know why.
As I've mentioned earlier the model does not underfit, but I think has a tendency to overfit on large amount of learning epochs


* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?


number of layers was reduced, last FC layer was removed due to overfit. number of neurons in convolution layers was increased, that gave good boost to accuracy.

If a well known architecture was chosen:
* What architecture was chosen?
http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf

* Why did you believe it would be relevant to the traffic sign application?
because it was made for this dataset

* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
fairly well, due to stochastic nature of NN and many hyper-params the result is not stable, and depends on current random seed.  

###Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![Stop][image4] ![Speed Limit 60][image5] ![No Entry][image6] 
![Speed limit (30km/h)][image7] ![Yield][image8]

The forth image might be difficult to classify because it's not well-aligned

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The code for making predictions on my final model is located in the 12th cell of the Ipython notebook.

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| Speed limit (60km/h)	| Speed limit (60km/h)							|
| No Entry				| No Entry										|
| Speed limit (30km/h)	| _Speed limit (80km/h)_						|
| Yield					| Yield											|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of 92.7%

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

##### Stop sign

|                      |                            |     
|:--------------------:|:--------------------------:| 
| __Stop__             | '1.0'						|
| Speed limit (30km/h) | '3.9417757889898654e-38'	|
| Speed limit (20km/h) | '0.0'						|
| Speed limit (50km/h) | '0.0'						|
| Speed limit (60km/h) | '0.0'						|

##### Speed limit (60km/h)

|                           |                           |     
|:-------------------------:|:-------------------------:| 
|__Speed limit (60km/h)__ 	| 0.9978376030921936 		|
|Speed limit (80km/h)       | 0.002162455813959241		|
|End of speed limit (80km/h)| 1.3976219699638393e-30	|
|Speed limit (20km/h)		| 0.0						|
|Speed limit (30km/h)		| 0.0						|
* all of options pretty close visually and by meaning

##### No entry

|                           |                           |     
|:-------------------------:|:-------------------------:| 
|__No entry__ | '1.0'|
|Speed limit (20km/h) | '0.0'|
|Speed limit (30km/h) | '0.0'|
|Speed limit (50km/h) | '0.0'|
|Speed limit (60km/h) | '0.0'|


##### Speed limit (30km/h)

|                           |                           |     
|:-------------------------:|:-------------------------:| 
|Speed limit (80km/h) | '1.0'|
|Go straight or right | '1.385089050837423e-36'|
|Speed limit (20km/h) | '0.0'|
|__Speed limit (30km/h)__ | '0.0'| 
|Speed limit (50km/h) | '0.0'|

model definitely classify this sign as speed limit, and even found pretty close (visually) 8<>3

##### Yield

|                           |                           |     
|:-------------------------:|:-------------------------:| 
| __Yield__ | '1.0' |
| Speed limit (20km/h) | '0.0' |
| Speed limit (30km/h) | '0.0' |
| Speed limit (50km/h) | '0.0' |
| Speed limit (60km/h) | '0.0' |


![probs][image9]

