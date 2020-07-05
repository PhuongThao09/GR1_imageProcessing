import cv2
import numpy as np
import matplotlib.pyplot as plt

# Reading the input image
img = cv2.imread('co-gian.jpg', 0)
retval, threshold = cv2.threshold(img, 90, 255, cv2.THRESH_BINARY_INV)
# Taking a matrix of size 5 as the kernel
kernel = np.ones((10, 10), np.uint8)

# The first parameter is the original image,
# kernel is the matrix with which image is
# convolved and third parameter is the number
# of iterations, which will determine how much

img_erosion = cv2.erode(threshold, kernel, iterations=1)
img_dilation = cv2.dilate(threshold, kernel, iterations=1)

f, axs = plt.subplots(1,2, figsize=(20,5))

axs[1].imshow(img_erosion)
axs[1].set_title('Erosion')
axs[0].imshow(img_dilation)
axs[0].set_title('Dilation')
cv2.imshow('Input', threshold)
# cv2.imshow('Erosion', img_erosion)
# cv2.imshow('Dilation', img_dilation)
plt.show()
cv2.waitKey(0)