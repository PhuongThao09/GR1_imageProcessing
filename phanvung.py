import cv2
import numpy as np
import matplotlib.pyplot as plt

# read the image
image = cv2.imread("anhdep.jpg")

# convert to RGB
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# reshape the image to a 2D array of pixels and 3 color values (RGB)
pixel_values = image.reshape((-1, 3))
# convert to float
pixel_values = np.float32(pixel_values)

# define stopping criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)

# number of clusters (K)
k = 3
_, labels, (centers) = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

# convert back to 8 bit values
centers = np.uint8(centers)

# flatten the labels array
labels = labels.flatten()

# convert all pixels to the color of the centroids
segmented_image = centers[labels.flatten()]

segmented_image = segmented_image.reshape(image.shape)
f, axs = plt.subplots(1,2, figsize=(10,5))
#plt.imshow(segmented_image)
axs[0].imshow(image)
axs[0].set_title('origin')
axs[1].imshow(segmented_image)
axs[1].set_title('segmented_image')
plt.show()
