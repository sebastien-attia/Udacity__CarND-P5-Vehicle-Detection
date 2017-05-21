**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/vehicle_non-vehicle.png
[image2]: ./output_images/car_HOG_visualization.png
[image3]: ./output_images/not_car_HOG_visualization.png
[image4]: ./output_images/find_cars.png
[image5]: ./examples/bboxes_and_heat.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[video1]: ./project_video.mp4

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

In order to have fewer false positives, I used:
- the decision_function() of the Linear SVM classifier, which gives the distance to the
  hyperplane, instead of using the function predict()

This algorithm is implemented in the 9th cell of the IPython notebook.






#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and...

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using...

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I decided to search random window positions at random scales all over the image and came up with this (ok just kidding I didn't actually ;):

![alt text][image3]

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image4]
---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]



---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  
