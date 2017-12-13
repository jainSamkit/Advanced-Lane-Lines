# Advanced-Lane-Lines
Finding the curved lane lines for self driving car
# Project 04 - Advanced Lane Finding
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

## README
The code I used for doing this project can be found in `project04.py` and `Project04.ipynb`. All the line numbers I refer to in this document is for `project04.py`. The following sections go into further detail about the specific points described in the [Rubric](https://review.udacity.com/#!/rubrics/571/view).

### Usage


**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./examples/undistort_output.png "Distorted"
[image2]: 
[image3]: ./test_images/test1.jpg "Road Transformed"
[image4]: ./examples/binary_combo_example.jpg "Binary Example"
[image5]: ./examples/warped_straight_lines.jpg "Warp Example"
[image6]: ./examples/color_fit_lines.jpg "Fit Visual"
[image7]: ./examples/example_output.jpg "Output"
[video1]: ./project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for camera calibration is contained in the lines 8-46 of my code in lane_lines.py. 

I prepared two lists for objpoints and imagepoints to store corresponding real world undistorted image points and distorted image points as referred in the image.The object points are similar to the rectangular meshgrid provided by the opencv function as follows:
objp=np.ones((9*6,3),np.float32)
objp[:,:2]=np.mgrid[0:9,0:6].T.reshape(-1,2)

First of all the objp is constructed which is array with height 9 and width 6 with three dimensions.After this the mgrid function constructs the integer coordinates corresponding to the image dimensions i.e height and width.

A.) ret,corners=cv2.findChessboardCorners(gray,(9,6),None)
The function at A is then used to find out the chessboard corners on the grayscale images.The imagepoints here are stored corresponding to the objectpoints here.

Then the cv2.calibrateCamera function is used to find the undistortion matrix and corresponding coefficients.They have been used in the line 109 of my code.Line 115 of the file is used to undistort the image given the undistortion matrix. 

There were some issues while in the calibration step which is to be discussed later.

![Undistorted Image1][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

I have provided an image of an undistorted image that I further used to work upon for further transforms and wapring.
![alt text][image2]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.

I used the the combinations of different channels of the HLS image and the gradient thresholds to identify and depict the lane lines perfectly.The function gradient_color_thresh below shows how I used the gradient and color thresholds.
```python 
def gradient_color_thresh(image):
    ksize=3
    image=undistort(image)
    gradx = abs_sobel_thresh(image, orient='x', thresh=(20, 200))
    grady = abs_sobel_thresh(image, orient='y', thresh=(20, 200))
    mag_binary = mag_thresh(image, mag_thresh=(20, 200))

    dir_binary = dir_threshold(image, thresh=(0.7, 1.3))
    color_binary=color_space(image,thresh=(100,255))
    
    combined = np.zeros_like(dir_binary)
    combined[(color_binary==1)|((gradx == 1)& (grady == 1)) |(mag_binary==1) &(dir_binary==1)] = 1
    kernel = np.ones((3,3),np.uint8)
    morph_image=combined[600:,:950]
    morph_image = cv2.morphologyEx(morph_image, cv2.MORPH_OPEN, kernel)
    combined[600:,:950]=morph_image
    return combined
```

#### Color Channels:
I used S channel to sepertate out the highly staurated i.e. the ones that have more brightness and not the background to filter out the lane lines.However,the S channel picked up the lane lines but there was extra noise appearing as well especially when shadows of the trees appeared on the roads.L channel represent the amount of light and dark pixels in an image.Since the shadows are dark I filtered out thr L channel pixels with values lower than 80.This mostly removed the shadows from the images.The python code below depicts this:

```python
def color_space(image,thresh=(170,255)):
    hls=cv2.cvtColor(image,cv2.COLOR_RGB2HLS)
    l_channel=hls[:,:,1]
    
    s_channel=hls[:,:,2]
    s_binary=np.zeros_like(s_channel)
    s_binary[(s_channel>=thresh[0]) & (s_channel<=thresh[1])& (l_channel>=80) ]=1
    color_output=np.copy(s_binary)
    return color_output
```

#### Gradient Thresholds:
I also used various gradient thresholds to seperate out the lane lines from the image.I also experimented with Roberts operator apart from the Sobel but Sobel came out to be a lot smoother while detecting the edges .Apart from using Sobel operators in X and Y directions ,I also used magnitude of the gradient and direction to seperate out the redundant lines from the image as well.
The python code is attached below:

```python
def dir_threshold(img,  thresh=(0.7,1.3)):
    gray=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    sobelx=np.absolute(cv2.Sobel(gray,cv2.CV_64F,1,0))
    sobely=np.absolute(cv2.Sobel(gray,cv2.CV_64F,0,1))
    dir_=np.arctan2(sobely,sobelx)
    sx_binary = np.zeros_like(gray)
    sx_binary[(dir_>=thresh[0]) &(dir_<=thresh[1])]=1
    binary_output=sx_binary
    return binary_output
   ```

![alt text][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for perpective tranform is done by the function perpective_transform in the line 242-249 in lane_lines.py. It takes an input image and applies warps it using source and destination points.

```python
    def perspective_transform(image):
    src=np.float32([[195,720],[590,460],[700,460],[1120,720]])
    dst=np.float32([[350,720],[410,0],[970,0],[1000,720]])
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    img_size=(image.shape[1],image.shape[0])
    warped = cv2.warpPerspective(image, M, img_size, flags=cv2.INTER_LINEAR)
    return warped
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 195, 720      | 350, 720        | 
| 590,460      | 410,0     |
| 700,460     | 970,0     |
| 1120,720    | 1000,720       |

I checked if my perpective transform was warping an image correctly by verifying it on the 
![alt text][image4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Then I did some other stuff and fit my lane lines with a 2nd order polynomial kinda like this:

![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines # through # in my code in `my_other_file.py`

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines # through # in my code in `yet_another_file.py` in the function `map_lane()`.  Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  



[//]: # (References)

[image1]: ./figures/undistort.png "Undistorted images"
[image3]: ./figures/threshold_edges.png "Thresholded Image"
[image4]: ./figures/perpective.png "Warp Example"
[image6]: ./figures/lane_mask.png "Lane masks"
[image7]: ./figures/highlighted_lane.png "Output"
[video1]: ./project_videos_output/result_video.mp4 "Project video output"
