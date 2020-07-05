import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('cat.jpg')
kernel = np.ones((5,5), np.float32)/25
blur_img = cv2.filter2D(img, -1, kernel)

f, axs = plt.subplots(1,2, figsize=(10,5))
axs[0].imshow(img)
axs[1].imshow(blur_img)
plt.show()