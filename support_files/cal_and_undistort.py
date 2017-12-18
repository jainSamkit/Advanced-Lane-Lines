
# coding: utf-8

# In[1]:


import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
get_ipython().magic('matplotlib inline')
import glob
get_ipython().magic('matplotlib qt')


# In[8]:


def calib(path):
    images=glob.glob('./camera_cal/calibration*.jpg')
    objpoints=[]
    imgpoints=[]
    
    objp=np.ones((9*6,3),np.float32)
    objp[:,:2]=np.mgrid[0:9,0:6].T.reshape(-1,2)
    for image in images:
        img=mpimg.imread(image)
        gray=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
        ret,corners=cv2.findChessboardCorners(gray,(9,6),None)
    
    if ret==True:
        imgpoints.append(corners)
        objpoints.append(objp)
    
    objp=np.ones((6*5,3),np.float32)
    objp[:,:2]=np.mgrid[0:6,0:5].T.reshape(-1,2)
    
    extra_images=['./camera_cal/calibration4.jpg']
    
    for image in extra_images:
        img=mpimg.imread(image)
        gray=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
        ret,corners=cv2.findChessboardCorners(gray,(6,5),None)
        if ret==True:
            imgpoints.append(corners)
            objpoints.append(objp)
            
            
    objp=np.ones((9*5,3),np.float32)
    objp[:,:2]=np.mgrid[0:9,0:5].T.reshape(-1,2)

    extra_images=['./camera_cal/calibration1.jpg']
    for image in extra_images:
        img=mpimg.imread(image)
        gray=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
        ret,corners=cv2.findChessboardCorners(gray,(9,5),None)
    
        if ret==True:
            imgpoints.append(corners)
            objpoints.append(objp)
            
    objp=np.ones((7*5,3),np.float32)
    objp[:,:2]=np.mgrid[0:7,0:5].T.reshape(-1,2)

    extra_images=['./camera_cal/calibration5.jpg']
    for image in extra_images:
        img=mpimg.imread(image)
        gray=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
        ret,corners=cv2.findChessboardCorners(gray,(7,5),None)
    
        if ret==True:
            imgpoints.append(corners)
            objpoints.append(objp)
            
    ref_image=images[0]
    ref_image=mpimg.imread(ref_image)
    gray=cv2.cvtColor(ref_image,cv2.COLOR_RGB2GRAY)
    
    return cv2.calibrateCamera(objpoints, imgpoints, gray.T.shape, None, None,flags=cv2.CALIB_USE_INTRINSIC_GUESS)


# In[11]:


def undistort(img,mtx,dst):
    return cv2.undistort(img, mtx, dst, None, mtx)

