######################################################################
### VC i PSIV                                                      ###
### Lab 0 (basat en material de Gemma Rotger)                      ###
######################################################################


# Hello! Welcome to the computer vision LAB. 
import time
import cv2
import numpy as np
from matplotlib import pyplot as plt


## PROBLEM 1 (+0.5) --------------------------------------------------
# DONE. READ THE CAMERAMAN IMAGE. 

cameraman_img = cv2.imread('./img/cameraman.jpg', cv2.IMREAD_GRAYSCALE)

## PROBLEM 2 (+0.5) --------------------------------------------------
# DONE: SHOW THE CAMERAMAN IMAGE

plt.imshow(cameraman_img, 'gray')
plt.show()

## PROBELM 3 (+2.0) --------------------------------------------------
# DONE. Negative efect using a double for instruction

t=time.time()
neg_cameraman_img = np.empty(cameraman_img.shape)
print(neg_cameraman_img.shape)
for row in range(cameraman_img.shape[0]):
    for col in range(cameraman_img.shape[1]):
        neg_cameraman_img[row, col] = 255 - cameraman_img[row, col]
elapsed=time.time()-t
print('Elapsed time is '+str(elapsed)+' seconds')
plt.figure("Naive (double for) calculation")
plt.imshow(neg_cameraman_img,'gray')
plt.show()

# DONE. Negative efect using a vectorial instruction

t=time.time()
neg_cameraman_img = np.empty(cameraman_img.shape)
neg_cameraman_img = 255 - cameraman_img
elapsed=time.time()-t
print('Elapsed time is '+str(elapsed)+' seconds')
plt.figure("Vectorial calculation")
plt.imshow(neg_cameraman_img,'gray')
plt.show()

# You should see that results in figures 1 and 2 are the same but times
# are much different.

## PROBLEM 4 (+2.0) --------------------------------------------------
# DONE. Give some color (red, green or blue)

r = cameraman_img
g = neg_cameraman_img
b = cameraman_img

colored_cameraman_img = np.empty((cameraman_img.shape[0], cameraman_img.shape[1],3), np.uint8)
colored_cameraman_img[:,:,0], colored_cameraman_img[:,:,1], colored_cameraman_img[:,:,2] = r, g, b
plt.figure("Colored image (normal, neg, normal)")
plt.imshow(colored_cameraman_img)
plt.show()

colored_cameraman_img = np.dstack((r,g,b))
plt.figure("Colored image (normal, neg, normal) - Using dstack() function")
plt.imshow(colored_cameraman_img)
plt.show()

## PROBLEM 5 (+1.0) --------------------------------------------------

cv2.imwrite("./img/generated_colored_cameraman_img.png", colored_cameraman_img)
cv2.imwrite("./img/generated_colored_cameraman_img.tif", colored_cameraman_img)
cv2.imwrite("./img/generated_colored_cameraman_img.jpg", colored_cameraman_img)
cv2.imwrite("./img/generated_colored_cameraman_img.bmp", colored_cameraman_img)

raw_size_kbytes = colored_cameraman_img.size/1024
print(raw_size_kbytes)

## PROBLEM 6 (+1.0) --------------------------------------------------

lin128 = cameraman_img[128, :]
plt.figure('Row 128 plot - original image')
plt.plot(lin128, 'k')
plt.plot(np.full(cameraman_img.shape[1], np.mean(cameraman_img[128, :])), '--m')
plt.show()

print(colored_cameraman_img.shape)
lin128rgb = colored_cameraman_img[128,:,:]
print(lin128rgb.shape)
plt.plot(lin128rgb[:,0], 'r')
plt.plot(lin128rgb[:,1], 'g')
plt.plot(lin128rgb[:,2], 'b')
plt.plot(np.full(cameraman_img.shape[1], np.mean(lin128rgb[:,:])), '--m')
plt.show()

## PROBLEM 7 (+2) ----------------------------------------------------

# TODO. Compute the histogram.
t=time.time()
# hist,bins = np.histogram...
elapsed=time.time()-t
print('Elapsed time is '+str(elapsed)+' seconds')
# plt.plot ...
# plt.show()

t=time.time()
# h=zeros(1,256);
# for ...
# plt.plot ...
# plt.show()
elapsed=time.time()-t
print('Elapsed time is '+str(elapsed)+' seconds')

## PROBLEM 8 Binarize the image text.png (+1) ------------------------

# TODO. Read the image
# imtext = ...
# plt.imshow(imtext)
# plt.show()
# hist,bins = np.histogram...
# plt.plot...
# plt.show()

# TODO. Define 3 different thresholds
# th1 = ...
# th2 = ...
# th3 = ...

# TODO. Apply the 3 thresholds 5 to the image
# threshimtext1 = ...
# threshimtext2 = ...
# threshimtext3 = ...

# TODO. Show the original image and the segmentations in a subplot
fig, ax = plt.subplots(nrows=2, ncols=3);
ax[0,0].remove()
ax[0,1].imshow(imtext)
ax[0,1].set_title('Original image')
ax[0,2].remove()
ax[1,0].imshow(threshimtext1)
ax[1,1].imshow(threshimtext2)
ax[1,2].imshow(threshimtext3)
plt.show()


## THE END -----------------------------------------------------------
# Well done, you finished this lab! Now, remember to deliver it 
# properly on Caronte.

# File name:
# lab0_NIU.zip 
# (put matlab file lab0.m and python file lab0.py in the same zip file)
# Example lab0_1234567.zip

















