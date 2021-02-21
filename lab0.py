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

colored_cameraman_img = np.empty(
    (cameraman_img.shape[0], cameraman_img.shape[1],3),
    np.uint8)
colored_cameraman_img[:,:,0] = r
colored_cameraman_img[:,:,1] = g
colored_cameraman_img[:,:,2] = b
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
print("RAW size:", raw_size_kbytes, "kb")

## PROBLEM 6 (+1.0) --------------------------------------------------

lin128 = cameraman_img[127, :]
plt.figure('Row 128 plot - original image')
plt.plot(lin128, 'k')
plt.plot(np.full(cameraman_img.shape[1], np.mean(cameraman_img[128, :])), '--m')
plt.show()

lin128rgb = colored_cameraman_img[127,:,:]
lin128rgb_mean = np.mean(lin128rgb, axis=1)
plt.figure('Row 128 plot - colored image')
plt.plot(lin128rgb[:,0], 'r')
plt.plot(lin128rgb[:,1], 'g')
plt.plot(lin128rgb[:,2], 'b')
plt.plot(np.full(cameraman_img.shape[1], lin128rgb_mean), '--m')
plt.show()

## PROBLEM 7 (+2) ----------------------------------------------------

pict0004_img = cv2.imread('img/pict0004.png')
t22_img = cv2.imread('img/t22.jpg')

# DONE. Compute the histogram.
t=time.time()
plt.figure("cameraman.jpg histogram")
plt.hist(cameraman_img.flatten(), range(256))
plt.show()
plt.figure("pict0004.png histogram")
plt.hist(pict0004_img.flatten(), range(256))
plt.show()
plt.figure("t22.jpg histogram")
plt.hist(t22_img.flatten(), range(256))
plt.show()
elapsed=time.time()-t
print('Elapsed time is '+str(elapsed)+' seconds')

t=time.time()
values = np.zeros(256)
for pixel in cameraman_img.flatten():
    values[pixel] += 1
plt.figure("Self-calculated histogram of cameraman.jpg")
plt.bar(range(256), values)
plt.show()
values = np.zeros(256)
for pixel in pict0004_img.flatten():
    values[pixel] += 1
plt.figure("Self-calculated histogram of pict0004.png")
plt.bar(range(256), values)
plt.show()
values = np.zeros(256)
for pixel in t22_img.flatten():
    values[pixel] += 1
plt.figure("Self-calculated histogram of t22.jpg")
plt.bar(range(256), values)
plt.show()
elapsed=time.time()-t
print('Elapsed time is '+str(elapsed)+' seconds')

## PROBLEM 8 Binarize the image text.png (+1) ------------------------

# DONE. Read the image
alice_img = cv2.imread('img/alice.jpg')
alice_img = cv2.cvtColor(alice_img, cv2.COLOR_BGR2GRAY)
plt.figure("Alice.jpg image")
plt.imshow(alice_img, 'gray')
plt.show()
plt.figure("Alice.jpg image histogram")
plt.hist(alice_img.flatten(), range(256))
plt.show()

# DONE. Define 3 different thresholds
th1 = 85
th2 = 195
th3 = 244

# DONE. Apply the 3 thresholds 5 to the image
threshimtext1 = alice_img > th1
plt.figure("Alice image as binary - threshold 1")
plt.imshow(threshimtext1, 'gray')
plt.show()
threshimtext2 = alice_img > th2
plt.figure("Alice image as binary - threshold 2")
plt.imshow(threshimtext2, 'gray')
plt.show()
threshimtext3 = alice_img > th3
plt.figure("Alice image as binary - threshold 3")
plt.imshow(threshimtext3, 'gray')
plt.show()

# TODO. Show the original image and the segmentations in a subplot
fig, ax = plt.subplots(nrows=2, ncols=3)
fig.canvas.set_window_title('Alice original image and the binarized images') 
ax[0,0].remove()
ax[0,1].imshow(alice_img, 'gray')
ax[0,1].set_title('Original image')
ax[0,2].remove()
ax[1,0].imshow(threshimtext1, 'gray')
ax[1,1].imshow(threshimtext2, 'gray')
ax[1,2].imshow(threshimtext3, 'gray')
plt.tight_layout()
plt.show()


## THE END -----------------------------------------------------------
# Well done, you finished this lab! Now, remember to deliver it 
# properly on Caronte.

# File name:
# lab0_NIU.zip 
# (put matlab file lab0.m and python file lab0.py in the same zip file)
# Example lab0_1234567.zip

















