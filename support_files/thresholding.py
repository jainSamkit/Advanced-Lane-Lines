
# coding: utf-8

# In[4]:


import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
get_ipython().magic('matplotlib inline')
import glob
get_ipython().magic('matplotlib qt')


# In[5]:


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
    gray_image=cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
    l_channel=hls[:,:,1]
    s_channel=hls[:,:,2]
    s_binary=np.zeros_like(s_channel)
    
    _, gray_binary = cv2.threshold(gray_image.astype('uint8'), 150, 255, cv2.THRESH_BINARY)
    s_binary[(s_channel>=thresh[0]) & (s_channel<=thresh[1])&(l_channel>=80)]=1
    color_output=np.copy(s_binary)
    return color_output

def segregate_white_line(image,thresh=(200,255)):
    hls=cv2.cvtColor(image,cv2.COLOR_RGB2HLS)
    l_channel=hls[:,:,1]
    l_binary=np.zeros_like(l_channel)
    l_binary[((l_channel>=200)&(l_channel<=255))]=1
    return l_binary


# In[6]:


def gradient_color_thresh(image):
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
    white_line=segregate_white_line(image,thresh=(200,255))
    combined=(combined)|(white_line)
    return combined


# In[ ]:




