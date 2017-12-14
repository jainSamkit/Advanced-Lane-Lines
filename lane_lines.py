
# coding: utf-8

# In[5]:


import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
get_ipython().magic('matplotlib inline')
import glob
get_ipython().magic('matplotlib qt')


# In[6]:


images=glob.glob('camera_cal/calibration*.jpg')
objpoints=[]
imgpoints=[]


# In[7]:


objp=np.ones((9*6,3),np.float32)
objp[:,:2]=np.mgrid[0:9,0:6].T.reshape(-1,2)


# In[8]:


for image in images:
    img=mpimg.imread(image)
    gray=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    ret,corners=cv2.findChessboardCorners(gray,(9,6),None)
    
    if ret==True:
        imgpoints.append(corners)
        objpoints.append(objp)
        
        img=cv2.drawChessboardCorners(img,(9,6),corners,ret)
        plt.imshow(img)
    else:
        print(image)


# In[9]:


objp=np.ones((6*5,3),np.float32)
objp[:,:2]=np.mgrid[0:6,0:5].T.reshape(-1,2)
extra_images=['camera_cal/calibration4.jpg']
for image in extra_images:
    img=mpimg.imread(image)
    gray=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    ret,corners=cv2.findChessboardCorners(gray,(6,5),None)
    if ret==True:
        imgpoints.append(corners)
        objpoints.append(objp)


# In[10]:


objp=np.ones((7*5,3),np.float32)
objp[:,:2]=np.mgrid[0:7,0:5].T.reshape(-1,2)

extra_images=['camera_cal/calibration5.jpg']
for image in extra_images:
    img=mpimg.imread(image)
    gray=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    ret,corners=cv2.findChessboardCorners(gray,(7,5),None)
    
    if ret==True:
        imgpoints.append(corners)
        objpoints.append(objp)


# In[11]:


objp=np.ones((9*5,3),np.float32)
objp[:,:2]=np.mgrid[0:9,0:5].T.reshape(-1,2)

extra_images=['camera_cal/calibration1.jpg']
for image in extra_images:
    img=mpimg.imread(image)
    gray=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    ret,corners=cv2.findChessboardCorners(gray,(9,5),None)
    
    if ret==True:
        imgpoints.append(corners)
        objpoints.append(objp)


# In[12]:


ref_image=images[0]
ref_image=mpimg.imread(ref_image)
gray=cv2.cvtColor(ref_image,cv2.COLOR_RGB2GRAY)


# In[13]:


ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None,flags=cv2.CALIB_USE_INTRINSIC_GUESS)


# In[14]:


dst = cv2.undistort(ref_image, mtx, dist, None, mtx)


# In[15]:


straight_images=glob.glob('test_images/straight_lines*.jpg')
straight_images


# In[16]:


curved_images=glob.glob('test_images/test*.jpg')
curved_images


# In[17]:


def undistort(image):
    return cv2.undistort(image, mtx, dist, None, mtx)


# In[18]:


def abs_sobel_thresh(image,orient,  thresh=(20, 100)):
    gray=cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
    if orient=='x':
        sobelx=cv2.Sobel(gray,cv2.CV_64F,1,0)
        sobelx=np.absolute(sobelx)
        scaled_sobel=np.uint8(255*sobelx/np.max(sobelx))
        sx_binary=np.zeros_like(scaled_sobel)
        sx_binary[(scaled_sobel>=thresh[0]) & (scaled_sobel<=thresh[1])]=1
        binary_output=np.copy(sx_binary)
    if orient=='y':
        sobely=cv2.Sobel(gray,cv2.CV_64F,0,1)
        sobely=np.absolute(sobely)
        scaled_sobel=np.uint8(255*sobely/np.max(sobely))
        sx_binary=np.zeros_like(scaled_sobel)
        sx_binary[(scaled_sobel>=thresh[0]) & (scaled_sobel<=thresh[1])]=1
        binary_output=np.copy(sx_binary)
    return binary_output


def mag_thresh(img, mag_thresh=(20,150)):
    
    # Apply the following steps to img
    gray=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    sobelx=cv2.Sobel(gray,cv2.CV_64F,1,0)
    sobely=cv2.Sobel(gray,cv2.CV_64F,0,1)
    sobel=np.sqrt(np.square(sobelx)+np.square(sobely))
    scaled_sobel=np.uint8(255*sobel/np.max(sobel))
    
    t=sum((i > 150) &(i<200)  for i in scaled_sobel)
#     print(np.sum(t))
#     print(scaled_sobel.shape)
    binary_sobel=np.zeros_like(scaled_sobel)
    binary_sobel[(scaled_sobel>=mag_thresh[0]) & (scaled_sobel<=mag_thresh[1])]=1
#     print(mag_thresh[0],mag_thresh[1])
#     binary_sobel[(scaled_sobel>=20) & (scaled_sobel<=150)]=1
#     plt.imshow(binary_sobel)
#     plt.show()
    return binary_sobel

def dir_threshold(img,  thresh=(0.7,1.3)):
    gray=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    sobelx=np.absolute(cv2.Sobel(gray,cv2.CV_64F,1,0))
    sobely=np.absolute(cv2.Sobel(gray,cv2.CV_64F,0,1))
    dir_=np.arctan2(sobely,sobelx)
    sx_binary = np.zeros_like(gray)
#     print(thresh[0],thresh[1])
    sx_binary[(dir_>=thresh[0]) &(dir_<=thresh[1])]=1
    binary_output=sx_binary
    return binary_output

def color_space(image,thresh=(170,255)):
    hls=cv2.cvtColor(image,cv2.COLOR_RGB2HLS)
    l_channel=hls[:,:,1]
#     plt.imshow(l_channel)
#     plt.show()
    s_channel=hls[:,:,2]
#     plt.imshow(s_channel)
#     plt.show()
    s_binary=np.zeros_like(s_channel)
    s_binary[(s_channel>=thresh[0]) & (s_channel<=thresh[1])& (l_channel>=80) ]=1
    color_output=np.copy(s_binary)
    return color_output


# In[19]:


def gradient_color_thresh(image):
    ksize=3
    image=undistort(image)
    gradx = abs_sobel_thresh(image, orient='x', thresh=(20, 200))
    grady = abs_sobel_thresh(image, orient='y', thresh=(20, 200))
    # plt.imshow(gradx)
    # plt.show()
    # plt.imshow(grady)
    # plt.show()
    mag_binary = mag_thresh(image, mag_thresh=(20, 200))
    # plt.imshow(mag_binary)
    # plt.show()

    dir_binary = dir_threshold(image, thresh=(0.7, 1.3))
    color_binary=color_space(image,thresh=(100,255))
    
    combined = np.zeros_like(dir_binary)
    # combined[((gradx == 1)& (grady == 1)) |(color_binary==1)] = 1
    combined[(color_binary==1)|((gradx == 1)& (grady == 1)) |(mag_binary==1) &(dir_binary==1)] = 1
#     plt.imshow(combined)
#     plt.show()
    kernel = np.ones((3,3),np.uint8)
    morph_image=combined[600:,:950]
    # morph_image = cv2.morphologyEx(morph_image, cv2.MORPH_CLOSE, kernel)
    morph_image = cv2.morphologyEx(morph_image, cv2.MORPH_OPEN, kernel)
    # morph_image=cv2.erode(morph_image,kernel,iterations = 1)
    combined[600:,:950]=morph_image
    return combined


# In[20]:


def perspective_transform(image):
    src=np.float32([[195,720],[590,460],[700,460],[1120,720]])
    dst=np.float32([[350,720],[410,0],[970,0],[1000,720]])
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    img_size=(image.shape[1],image.shape[0])
    warped = cv2.warpPerspective(image, M, img_size, flags=cv2.INTER_LINEAR)
    return warped


# In[21]:


src=np.float32([[195,720],[590,460],[700,460],[1120,720]])
dst=np.float32([[350,720],[410,0],[970,0],[1000,720]])
Minv = cv2.getPerspectiveTransform(dst, src)


# In[22]:


def pipeline(warped,image,left_fit,right_fit,count):
    binary_warped=warped
    if count==0:
        # Assuming you have created a warped binary image called "binary_warped"
        # Take a histogram of the bottom half of the image
        histogram = np.sum(binary_warped[int(binary_warped.shape[0]/2):,:], axis=0)
        # Create an output image to draw on and  visualize the result
        out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0]/2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        # Choose the number of sliding windows
        nwindows = 9
        # Set height of windows
        window_height = np.int(binary_warped.shape[0]/nwindows)
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Current positions to be updated for each window
        leftx_current = leftx_base
        rightx_current = rightx_base
        # Set the width of the windows +/- margin
        margin = 100
        # Set minimum number of pixels found to recenter window
        minpix = 50
        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []

        # Step through the windows one by one
        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = binary_warped.shape[0] - (window+1)*window_height
            win_y_high = binary_warped.shape[0] - window*window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin
            # Draw the windows on the visualization image
            cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),
            (0,255,0), 2) 
            cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),
            (0,255,0), 2) 
            # Identify the nonzero pixels in x and y within the window
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
            (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
            (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:        
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

        # Concatenate the arrays of indices
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds] 
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds] 

        # Fit a second order polynomial to each
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)

        ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    #     out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    #     out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
    #     plt.imshow(out_img)
    #     plt.plot(left_fitx, ploty, color='yellow')
    #     plt.plot(right_fitx, ploty, color='yellow')
    #     plt.xlim(0, 1280)
    #     plt.ylim(720, 0)
        warp_zero = np.zeros_like(warped).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
        # Minv = cv2.getPerspectiveTransform(dst, src)
        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        newwarp = cv2.warpPerspective(color_warp, Minv, (image.shape[1], image.shape[0])) 
        # Combine the result with the original image
        result = cv2.addWeighted(image, 1, newwarp, 0.3, 0)
        ploty = np.linspace(0, 719, num=720)# to cover same y-range as image
        quadratic_coeff = 3e-4 # arbitrary quadratic coefficient
        # For each y position generate random x position within +/-50 pix
        # of the line base position in each case (x=200 for left, and x=900 for right)
        leftx = np.array([200 + (y**2)*quadratic_coeff + np.random.randint(-50, high=51) 
                                      for y in ploty])
        rightx = np.array([900 + (y**2)*quadratic_coeff + np.random.randint(-50, high=51) 
                                        for y in ploty])

        leftx = leftx[::-1]  # Reverse to match top-to-bottom in y
        rightx = rightx[::-1]  # Reverse to match top-to-bottom in y


        # Fit a second order polynomial to pixel positions in each fake lane line
        left_fit = np.polyfit(ploty, leftx, 2)
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fit = np.polyfit(ploty, rightx, 2)
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
        y_eval = np.max(ploty)
        left_curverad = ((1 + (2*left_fit[0]*y_eval + left_fit[1])**2)**1.5) / np.absolute(2*left_fit[0])
        right_curverad = ((1 + (2*right_fit[0]*y_eval + right_fit[1])**2)**1.5) / np.absolute(2*right_fit[0])
        
        ym_per_pix = 30/720 # meters per pixel in y dimension
        xm_per_pix = 3.7/700 # meters per pixel in x dimension

        # Fit new polynomials to x,y in world space
        left_fit_cr = np.polyfit(ploty*ym_per_pix, leftx*xm_per_pix, 2)
        right_fit_cr = np.polyfit(ploty*ym_per_pix, rightx*xm_per_pix, 2)
        # Calculate the new radii of curvature
        left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
        right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
        middle = (left_fitx[-1] + right_fitx[-1])//2
        veh_pos = image.shape[1]//2
        difference = (veh_pos - middle)*xm_per_pix # Positive if on right, Negative on left
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(result,'Left radius of curvature  = %.2f m'%(left_curverad),(50,50), font, 1,(255,255,255),2,cv2.LINE_AA)
        cv2.putText(result,'Right radius of curvature = %.2f m'%(right_curverad),(50,80), font, 1,(255,255,255),2,cv2.LINE_AA)
        cv2.putText(result,'Vehicle at : %.2f m %s of center'%(abs(difference), 'left_side' if difference < 0 else 'right_side'),(50,110),
                        font, 1,(255,255,255),2,cv2.LINE_AA)
        return result
    

# In[23]:


from moviepy.editor import VideoFileClip
from IPython.display import HTML


# In[24]:


left_fit=[]
right_fit=[]
count=0


# In[25]:


def process_image(image):
    undist_image=undistort(image)
#     gray_image=cv2.cvtColor(undist_image,cv2.COLOR_RGB2GRAY)
    thresh_image=gradient_color_thresh(undist_image)
    warped=perspective_transform(thresh_image)
    result=pipeline(warped,undist_image,left_fit,right_fit,count)
    return result


# In[1646]:


video_output = 'project_videos_output/result_video.mp4'
## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
## To do so add .subclip(start_second,end_second) to the end of the line below
## Where start_second and end_second are integer values representing the start and end of the subclip
## You may also uncomment the following line for a subclip of the first 5 seconds
clip1 = VideoFileClip("project_video.mp4")
# clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4")
video_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
get_ipython().magic('time video_clip.write_videofile(video_output, audio=False)')


# In[22]:


def pipeline1(warped,image,count):
    binary_warped=warped
    if count==0:
        # Assuming you have created a warped binary image called "binary_warped"
        # Take a histogram of the bottom half of the image
        histogram = np.sum(binary_warped[int(binary_warped.shape[0]/2):,:], axis=0)
        # Create an output image to draw on and  visualize the result
        out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0]/2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        # Choose the number of sliding windows
        nwindows = 9
        # Set height of windows
        window_height = np.int(binary_warped.shape[0]/nwindows)
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Current positions to be updated for each window
        leftx_current = leftx_base
        rightx_current = rightx_base
        # Set the width of the windows +/- margin
        margin = 100
        # Set minimum number of pixels found to recenter window
        minpix = 50
        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []

        # Step through the windows one by one
        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = binary_warped.shape[0] - (window+1)*window_height
            win_y_high = binary_warped.shape[0] - window*window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin
            # Draw the windows on the visualization image
            cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),
            (0,255,0), 2) 
            cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),
            (0,255,0), 2) 
            # Identify the nonzero pixels in x and y within the window
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
            (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
            (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:        
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

        # Concatenate the arrays of indices
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds] 
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds] 

        # Fit a second order polynomial to each
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)

        ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    #     out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    #     out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
    #     plt.imshow(out_img)
    #     plt.plot(left_fitx, ploty, color='yellow')
    #     plt.plot(right_fitx, ploty, color='yellow')
    #     plt.xlim(0, 1280)
    #     plt.ylim(720, 0)
        warp_zero = np.zeros_like(warped).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
        # Minv = cv2.getPerspectiveTransform(dst, src)
        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        newwarp = cv2.warpPerspective(color_warp, Minv, (image.shape[1], image.shape[0])) 
        # Combine the result with the original image
        result = cv2.addWeighted(image, 1, newwarp, 0.3, 0)
        ploty = np.linspace(0, 719, num=720)# to cover same y-range as image
        quadratic_coeff = 3e-4 # arbitrary quadratic coefficient
        # For each y position generate random x position within +/-50 pix
        # of the line base position in each case (x=200 for left, and x=900 for right)
        leftx = np.array([200 + (y**2)*quadratic_coeff + np.random.randint(-50, high=51) 
                                      for y in ploty])
        rightx = np.array([900 + (y**2)*quadratic_coeff + np.random.randint(-50, high=51) 
                                        for y in ploty])

        leftx = leftx[::-1]  # Reverse to match top-to-bottom in y
        rightx = rightx[::-1]  # Reverse to match top-to-bottom in y


        # Fit a second order polynomial to pixel positions in each fake lane line
        left_fit = np.polyfit(ploty, leftx, 2)
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fit = np.polyfit(ploty, rightx, 2)
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
        y_eval = np.max(ploty)
        left_curverad = ((1 + (2*left_fit[0]*y_eval + left_fit[1])**2)**1.5) / np.absolute(2*left_fit[0])
        right_curverad = ((1 + (2*right_fit[0]*y_eval + right_fit[1])**2)**1.5) / np.absolute(2*right_fit[0])
        
        ym_per_pix = 30/720 # meters per pixel in y dimension
        xm_per_pix = 3.7/700 # meters per pixel in x dimension

        # Fit new polynomials to x,y in world space
        left_fit_cr = np.polyfit(ploty*ym_per_pix, leftx*xm_per_pix, 2)
        right_fit_cr = np.polyfit(ploty*ym_per_pix, rightx*xm_per_pix, 2)
        # Calculate the new radii of curvature
        left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
        right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
        middle = (left_fitx[-1] + right_fitx[-1])//2
        veh_pos = image.shape[1]//2
        dx = (veh_pos - middle)*xm_per_pix # Positive if on right, Negative on left
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(result,'Left radius of curvature  = %.2f m'%(left_curverad),(50,50), font, 1,(255,255,255),2,cv2.LINE_AA)
        cv2.putText(result,'Right radius of curvature = %.2f m'%(right_curverad),(50,80), font, 1,(255,255,255),2,cv2.LINE_AA)
        cv2.putText(result,'Vehicle position : %.2f m %s of center'%(abs(dx), 'left' if dx < 0 else 'right'),(50,110),
                        font, 1,(255,255,255),2,cv2.LINE_AA)
        return result


# In[23]:


from matplotlib import pyplot


# In[34]:


k=0
for image in straight_images:
    image_string=image
    image=mpimg.imread(image)
    imstack = cv2.resize(image,(6000,4800))
    undist_image=undistort(image)
    thresh_image=gradient_color_thresh(undist_image)
    warped=perspective_transform(thresh_image)
    result=pipeline1(warped,undist_image,0)
    
    im = cv2.resize(undist_image,(6000,4800))
#     print(im.shape)
    imstack = np.hstack((imstack,im))
    im = cv2.resize(thresh_image,(6000,4800))
    im=np.dstack((im, im, im))*255
#     print(im.shape)
    imstack = np.hstack((imstack,im))
    im = cv2.resize(warped,(6000,4800))
    im=np.dstack((im, im, im))*255
    imstack = np.hstack((imstack,im))
    im = cv2.resize(result,(6000,4800))
    imstack = np.hstack((imstack,im))
    image_string = image_string.split("/")
    print('output_images/'+str(image_string[1]))
    cv2.imwrite('output_images/'+str(image_string[1]),imstack)
#     pyplot.imsave('output_images/'+str(image_string[1]),imstack )
    plt.imshow(imstack)
    plt.show()


# In[35]:


k=0
for image in curved_images:
    image_string=image
    image=mpimg.imread(image)
    
    imstack = cv2.cvtColor(cv2.resize(image,(6000,4800)),cv2.COLOR_BGR2RGB)
    undist_image=undistort(image)
    thresh_image=gradient_color_thresh(undist_image)
    warped=perspective_transform(thresh_image)
    result=pipeline1(warped,undist_image,0)
    
    im = cv2.cvtColor(cv2.resize(undist_image,(6000,4800)),cv2.COLOR_BGR2RGB)
    print(im.shape)
    imstack = np.hstack((imstack,im))
    im = cv2.resize(thresh_image,(6000,4800))
    im=np.dstack((im, im, im))*255
    print(im.shape)
    imstack = np.hstack((imstack,im))
    im = cv2.resize(warped,(6000,4800))
    im=np.dstack((im, im, im))*255
    imstack = np.hstack((imstack,im))
    im = cv2.resize(result,(6000,4800))
    imstack = np.hstack((imstack,im))
    image_string = image_string.split("/")
    print('output_images/'+str(image_string[1]))
    cv2.imwrite('output_images/'+str(image_string[1]),imstack)
#     pyplot.imsave('output_images/'+str(image_string[1]),imstack )
    plt.imshow(imstack)
    plt.show()


# In[92]:


from matplotlib import pyplot


# In[93]:


ksize=3
image=mpimg.imread(curved_images[2])
image=cv2.undistort(image, mtx, dist, None, mtx)
pyplot.imsave('figures/undistort.png',image)
gradx = abs_sobel_thresh(image, orient='x', thresh=(20, 200))
grady = abs_sobel_thresh(image, orient='y', thresh=(20, 200))
mag_binary = mag_thresh(image, mag_thresh=(20, 200))

dir_binary = dir_threshold(image, thresh=(0.7, 1.3))
color_binary=color_space(image,thresh=(100,255))


# In[94]:


combined = np.zeros_like(dir_binary)
# combined[((gradx == 1)& (grady == 1)) |(color_binary==1)] = 1
combined[(color_binary==1)|((gradx == 1)& (grady == 1)) |(mag_binary==1) &(dir_binary==1)] = 1
kernel = np.ones((3,3),np.uint8)
morph_image=combined[600:,:950]
# morph_image = cv2.morphologyEx(morph_image, cv2.MORPH_CLOSE, kernel)
morph_image = cv2.morphologyEx(morph_image, cv2.MORPH_OPEN, kernel)
# morph_image=cv2.erode(morph_image,kernel,iterations = 1)
combined[600:,:950]=morph_image

# combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel)
plt.imshow(combined)
plt.show()


# In[96]:


pyplot.imsave('figures/threshold_edges.png',combined)


# In[97]:


src=np.float32([[195,720],[590,460],[700,460],[1120,720]])
dst=np.float32([[350,720],[410,0],[970,0],[1000,720]])
M = cv2.getPerspectiveTransform(src, dst)
Minv = cv2.getPerspectiveTransform(dst, src)
img_size=(combined.shape[1],combined.shape[0])
warped = cv2.warpPerspective(combined, M, img_size, flags=cv2.INTER_LINEAR)


# In[98]:


plt.imshow(warped)
plt.show()
pyplot.imsave('figures/perpective.png',warped)


# In[99]:


# # window settings
# window_width = 50 
# window_height = 80 # Break image into 9 vertical layers since image height is 720
# margin = 50 # How much to slide left and right for searching

# def window_mask(width, height, img_ref, center,level):
#     output = np.zeros_like(img_ref)
#     output[int(img_ref.shape[0]-(level+1)*height):int(img_ref.shape[0]-level*height),max(0,int(center-width/2)):min(int(center+width/2),img_ref.shape[1])] = 1
#     return output

# def find_window_centroids(image, window_width, window_height, margin):
    
#     window_centroids = [] # Store the (left,right) window centroid positions per level
#     window = np.ones(window_width) # Create our window template that we will use for convolutions
    
#     # First find the two starting positions for the left and right lane by using np.sum to get the vertical image slice
#     # and then np.convolve the vertical image slice with the window template 
    
#     # Sum quarter bottom of image to get slice, could use a different ratio
#     l_sum = np.sum(image[int(3*image.shape[0]/4):,:int(image.shape[1]/2)], axis=0)
#     l_center = np.argmax(np.convolve(window,l_sum))-window_width/2
#     r_sum = np.sum(image[int(3*image.shape[0]/4):,820:], axis=0)
#     r_center = np.argmax(np.convolve(window,r_sum))-window_width/2+820

# #     r_center = np.argmax(np.convolve(window,r_sum))-window_width/2+int(image.shape[1]/2)
# #     print(np.convolve(window,r_sum))
#     # Add what we found for the first layer
#     window_centroids.append((l_center,r_center))
    
#     # Go through each layer looking for max pixel locations
#     for level in range(1,(int)(image.shape[0]/window_height)):
#         # convolve the window into the vertical slice of the image
#         image_layer = np.sum(image[int(image.shape[0]-(level+1)*window_height):int(image.shape[0]-level*window_height),:], axis=0)
#         conv_signal = np.convolve(window, image_layer)
#         # Find the best left centroid by using past left center as a reference
#         # Use window_width/2 as offset because convolution signal reference is at right side of window, not center of window
#         offset = window_width/2
#         l_center=int((np.mean(window_centroids,axis=0))[0])
#         l_min_index = int(max(l_center+offset-margin,0))
#         l_max_index = int(min(l_center+offset+margin,image.shape[1]))
#         l_center = np.argmax(conv_signal[l_min_index:l_max_index])+l_min_index-offset
#         # Find the best right centroid by using past right center as a reference
#         r_center=int((np.mean(window_centroids,axis=0))[1])
#         r_min_index = int(max(r_center+offset-margin,0))
#         r_max_index = int(min(r_center+offset+margin,image.shape[1]))
#         r_center = np.argmax(conv_signal[r_min_index:r_max_index])+r_min_index-offset
#         # Add what we found for that layer
#         window_centroids.append((l_center,r_center))

#     return window_centroids

# window_centroids = find_window_centroids(warped, window_width, window_height, margin)

# # If we found any window centers
# if len(window_centroids) > 0:

#     # Points used to draw all the left and right windows
#     l_points = np.zeros_like(warped)
#     r_points = np.zeros_like(warped)

#     # Go through each level and draw the windows 	
#     for level in range(0,len(window_centroids)):
#         # Window_mask is a function to draw window areas
# 	    l_mask = window_mask(window_width,window_height,warped,window_centroids[level][0],level)
# 	    r_mask = window_mask(window_width,window_height,warped,window_centroids[level][1],level)
# 	    # Add graphic points from window mask here to total pixels found 
# 	    l_points[(l_points == 255) | ((l_mask == 1) ) ] = 255
# 	    r_points[(r_points == 255) | ((r_mask == 1) ) ] = 255

#     # Draw the results
#     template = np.array(r_points+l_points,np.uint8) # add both left and right window pixels together
#     zero_channel = np.zeros_like(template) # create a zero color channel
#     template = np.array(cv2.merge((zero_channel,template,zero_channel)),np.uint8) # make window pixels green
#     warpage= np.dstack((warped, warped, warped))*255 # making the original road pixels 3 color channels
#     output = cv2.addWeighted(warpage, 1, template, 1.0, 0.0) # overlay the orignal road image with window results
# # If no window centers found, just display orginal road image
# else:
#     output = np.array(cv2.merge((warped,warped,warped)),np.uint8)

# # Display the final results
# plt.imshow(output)
# plt.title('window fitting results')
# plt.show()


# In[100]:


binary_warped=warped

# Assuming you have created a warped binary image called "binary_warped"
# Take a histogram of the bottom half of the image
histogram = np.sum(binary_warped[int(binary_warped.shape[0]/2):,:], axis=0)
# Create an output image to draw on and  visualize the result
out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
# Find the peak of the left and right halves of the histogram
# These will be the starting point for the left and right lines
midpoint = np.int(histogram.shape[0]/2)
leftx_base = np.argmax(histogram[:midpoint])
rightx_base = np.argmax(histogram[midpoint:]) + midpoint

# Choose the number of sliding windows
nwindows = 9
# Set height of windows
window_height = np.int(binary_warped.shape[0]/nwindows)
# Identify the x and y positions of all nonzero pixels in the image
nonzero = binary_warped.nonzero()
nonzeroy = np.array(nonzero[0])
nonzerox = np.array(nonzero[1])
# Current positions to be updated for each window
leftx_current = leftx_base
rightx_current = rightx_base
# Set the width of the windows +/- margin
margin = 100
# Set minimum number of pixels found to recenter window
minpix = 50
# Create empty lists to receive left and right lane pixel indices
left_lane_inds = []
right_lane_inds = []

# Step through the windows one by one
for window in range(nwindows):
    # Identify window boundaries in x and y (and right and left)
    win_y_low = binary_warped.shape[0] - (window+1)*window_height
    win_y_high = binary_warped.shape[0] - window*window_height
    win_xleft_low = leftx_current - margin
    win_xleft_high = leftx_current + margin
    win_xright_low = rightx_current - margin
    win_xright_high = rightx_current + margin
    # Draw the windows on the visualization image
    cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),
    (0,255,0), 2) 
    cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),
    (0,255,0), 2) 
    # Identify the nonzero pixels in x and y within the window
    good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
    (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
    good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
    (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
    # Append these indices to the lists
    left_lane_inds.append(good_left_inds)
    right_lane_inds.append(good_right_inds)
    # If you found > minpix pixels, recenter next window on their mean position
    if len(good_left_inds) > minpix:
        leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
    if len(good_right_inds) > minpix:        
        rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

# Concatenate the arrays of indices
left_lane_inds = np.concatenate(left_lane_inds)
right_lane_inds = np.concatenate(right_lane_inds)

# Extract left and right line pixel positions
leftx = nonzerox[left_lane_inds]
lefty = nonzeroy[left_lane_inds] 
rightx = nonzerox[right_lane_inds]
righty = nonzeroy[right_lane_inds] 

# Fit a second order polynomial to each
left_fit = np.polyfit(lefty, leftx, 2)
right_fit = np.polyfit(righty, rightx, 2)


# In[101]:


ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
plt.imshow(out_img)
plt.plot(left_fitx, ploty, color='yellow')
plt.plot(right_fitx, ploty, color='yellow')
plt.xlim(0, 1280)
plt.ylim(720, 0)


# In[105]:


out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
window_img = np.zeros_like(out_img)
# Color in left and right line pixels
out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

# Generate a polygon to illustrate the search window area
# And recast the x and y points into usable format for cv2.fillPoly()
left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, 
                              ploty])))])
left_line_pts = np.hstack((left_line_window1, left_line_window2))
right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, 
                              ploty])))])
right_line_pts = np.hstack((right_line_window1, right_line_window2))

# Draw the lane onto the warped blank image
cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
plt.imshow(result)
plt.plot(left_fitx, ploty, color='yellow')
plt.plot(right_fitx, ploty, color='yellow')
plt.xlim(0, 1280)
plt.ylim(720, 0)


# In[106]:


pyplot.imsave('figures/Lane_mask.png',result)


# In[107]:


warp_zero = np.zeros_like(warped).astype(np.uint8)
color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
# Minv = cv2.getPerspectiveTransform(dst, src)
# Recast the x and y points into usable format for cv2.fillPoly()
pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
pts = np.hstack((pts_left, pts_right))

# Draw the lane onto the warped blank image
cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

# Warp the blank back to original image space using inverse perspective matrix (Minv)
newwarp = cv2.warpPerspective(color_warp, Minv, (image.shape[1], image.shape[0])) 
# Combine the result with the original image
result = cv2.addWeighted(image, 1, newwarp, 0.3, 0)
plt.imshow(result)
pyplot.imsave('figures/highlighted_lane.png',result)


# In[ ]:





# In[ ]:




