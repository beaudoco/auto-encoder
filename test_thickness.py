import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt

def display(array1, array2):
    """
    Displays ten random images from each one of the supplied arrays.
    """

    n = 10

    indices = np.random.randint(len(array1), size=n)
    images1 = array1[indices, :]
    images2 = array2[indices, :]

    plt.figure(figsize=(20, 4))
    for i, (image1, image2) in enumerate(zip(images1, images2)):
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(image1.reshape(300, 300))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(image2.reshape(300, 300))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    plt.show()

for filename in glob.glob('USPTO-50K-IMAGES-SRC-TEST/*'):
    mask = np.reshape(np.load(filename), [300, 300, 1])
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
dilate = cv2.erode(mask, kernel, iterations=1)
cv2.imshow('image',mask)
cv2.waitKey(0)
# display(mask, dilate)