import cv2
import numpy as np

cap = cv2.imread("lena.jpg")

hsv = cv2.cvtColor(cap, cv2.COLOR_BGR2HSV)

lower_red = np.array([30, 150, 50])
upper_red = np.array([255, 255, 180])

mask = cv2.inRange(hsv, lower_red, upper_red)
res = cv2.bitwise_and(cap, cap, mask=mask)

cv2.imshow('Original', cap)
edges = cv2.Canny(cap, 100, 200)
cv2.imshow('Edges', edges)
cv2.waitKey()