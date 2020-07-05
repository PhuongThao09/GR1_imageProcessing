import cv2
import numpy as np

img = cv2.imread('mudo.jpg')
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

lower_red = np.array([30, 150, 50])
upper_red = np.array([255, 255, 180])

mask = cv2.inRange(hsv, lower_red, upper_red)
res = cv2.bitwise_and(img, img, mask=mask)

kernel = np.ones((15,15),np.float32)/225
smoothed = cv2.filter2D(res, -1, kernel)
cv2.imshow('Original', img)
cv2.imshow('Averaging', smoothed)
blur = cv2.GaussianBlur(res, (15,15),0)
cv2.imshow('Gaussian Blurring',blur)
cv2.waitKey()