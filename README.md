# Vehicle Detection
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)


**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.
* Discussion

[//]: # (Image References)
[image1]: ./output_images/vehicle_non-vehicle.png
[image2]: ./output_images/car_HOG_visualization.png
[image3]: ./output_images/not_car_HOG_visualization.png
[image4]: ./output_images/find_cars.png
[image5]: ./output_images/find_cars-raw_boxes.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[video1]: ./project.mp4

---

### Histogram of Oriented Gradients (HOG)

#### 1. HOG features from the training images.

The code for this step is contained in the 2nd code cell of the IPython notebook.  

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![Vehicle and non-vehicle images][image1]


Here is a HOG visualization of the image of a car, afetr converting the image in
gray and with the HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(4, 4)`:

![Car HOG visualization][image2]

### 2. Training of a linear SVM classifier

In the 8th cell of the notebook, I trained a linear SVM classifier, using :
- the HOG features and,
- the color features (spatial features and histogram features).

After playing with the HOG parameters and the color features parameters,
the best test accuracy I obtained is 98.9 % on a training set of 28416 samples (around
  half cars and half not cars).

The parameters I retained are:
- color space: `YCrCb`
- HOG parameters: `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(4, 4)`
- color features: `spatial_size=(32, 32)`, `hist_bins=32`
The total number of features is 13968.

### 3. Sliding window and image scaling

As vehicle to detect can be very close or very far, the size of the sliding window to
search a vehicle must vary. Another way of achieving this is to keep the same size
for the sliding window and to resize the image (using a scale factor).

This algorithm is implemented in the 12th cell of the IPython notebook.

In order to have fewer false positives, I used:
- the decision_function() of the Linear SVM classifier, which gives the distance to the
  hyperplane, instead of using the function predict(); predict() returns 1 when
  the decision_function() returns a value > 0. Here the trigger is 0.75.
- the result from the decision_function() is used to build the heat map; more this
  number is high, more this point is positevely far from the hyperplane "separation"
  (meaning this point is surely a car) and more its contribution to the heatmap is important.
- the region of interest is limited by setting 3 search boxes having the following
  parameters:
  format: `(xmin, xmax, ymin, ymax)`,
  `((0, 500, 500, 680), (200, 1200, 400, 500), (850, None, 380, 680))`

This algorithm is implemented in the 19th cell of the IPython notebook.

Here is an example of the sliding window with the scaling of the image:

![Car detected][image5]

### 4. The video stream pipeline

The video stream keeps track of the 5 last frames.
A heatmap is created and each box is marked by adding the value returned by the
function decision_function() of the classifier. In theory, higher is this value,
more we are sure that the classifier has identified a car in the box.
Then, the heatmap is thresholded to remove part of false positives.

I obtained the following video:
![Project video][video1]

### 5. Estimation of the bounding for the vehicles detected

The package scipy.ndimage.measurements.label is used to find adjacent boxes.
Then, the box is defined by the min/max x and y of adjacent boxes, to define a
encompassing box.

The estimation of the bounding is done in the 13th cell of the IPython notebook.

### 6. Discussion

I faced two main problems in this project:
- removing false positives,
- to have a relative stable boxing of the surrounding cars.

The current pipeline defines a region of interest to remove a big part of the false
positives. It is the easy way in this project to have a reasonable project video.
But it would not be acceptable in a real life project, as surrounding vehicles
should be detected, wherever they are.

The following options can be investigated to improve the accuracy of the vehicle detection:
- instead of using HOG features extraction + SVM to identify if a part of the image
  is a vehicle, we could use a Convolutional Neural Network as classifier and improve
  its accuracy; A CNN can not only identify a car, but it can identify too a pedestrian,
  a bicycle ... with a high accuracy.
- having a rear or a side camera could provide valuable information; a object
  coming from behind or from the side could be identified and could be tracked
  with algorithm like Kalman Filter.
