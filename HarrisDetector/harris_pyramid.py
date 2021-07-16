import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np
import scipy.ndimage.filters as filters


def pyramid(image, blur_nbr, k, sigma):
    res_shape = (blur_nbr, image.shape[0], image.shape[1], 3)
    res_img = np.zeros(res_shape)
    fimage = np.float64(image)
    prev_img = cv.GaussianBlur(fimage, (0, 0), sigmaX=sigma, sigmaY=sigma)
    
    for i in range(0, blur_nbr - 1):
        sigma = k*sigma
        img = cv.GaussianBlur(prev_img, (0, 0), sigmaX=sigma, sigmaY=sigma)
        res_img[i, :, :] = img - prev_img
        prev_img = img

    return res_img


def find_max(image, size, threshold):   # size - rozmiar maski filtra maksymalnego
    data_max = filters.maximum_filter(image, size)
    maxima = (image == data_max)
    diff = image > threshold
    maxima[diff == 0] = 0
    return np.nonzero(maxima)


i1 = cv.imread('images/fontanna1.jpg')
pow1 = cv.imread('images/fontanna_pow.jpg')

MASK_SIZE = 7

py1 = pyramid(i1, 5, 1.26, 1.6)
max_i1 = find_max(py1, MASK_SIZE, 0.2)
x1 = max_i1[2]
y1 = max_i1[1]

py2 = pyramid(pow1, 10, 1.26, 1.6)
max_pow1 = find_max(py2, MASK_SIZE, 0.2)
x2 = max_pow1[2]
y2 = max_pow1[1]

plt.figure(figsize=(12, 7))
for k in range(0, py1.shape[0] - 1):
    plt.subplot(2, 3, k+1)
    plt.imshow(py1[k, :, :])
    plt.axis('off')
    plt.plot(max_i1[2][max_i1[0] == k], max_i1[1][max_i1[0] == k], '*', color='m')
    plt.title('Difference:' + str(k+1))

plt.figure(figsize=(12, 9))
for k in range(0, py2.shape[0] - 1):
    plt.subplot(3, 4, k + 1)
    plt.imshow(py2[k, :, :])
    plt.axis('off')
    plt.plot(max_pow1[2][max_pow1[0] == k], max_pow1[1][max_pow1[0] == k], '*', color='m')
    plt.title('Difference:' + str(k + 1))
plt.show()
