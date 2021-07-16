import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage.filters as filters

MASK_SIZE = 7
THRESH = 0.5
K = 0.05

f1 = cv.imread('images/fontanna1.jpg', 0)
f2 = cv.imread('images/fontanna2.jpg', 0)

b1 = cv.imread('images/budynek1.jpg', 0)
b2 = cv.imread('images/budynek2.jpg', 0)


def calculate_H(image, MASK_SIZE):

    sobelx = cv.Sobel(image, cv.CV_32F, 1, 0, ksize=MASK_SIZE)
    sobely = cv.Sobel(image, cv.CV_32F, 0, 1, ksize=MASK_SIZE)

    sobelx2 = sobelx * sobelx
    sobely2 = sobely * sobely
    sobelxy = sobelx * sobely

    gaussianx2 = cv.GaussianBlur(sobelx2, (MASK_SIZE, MASK_SIZE), 0)
    gaussianxy = cv.GaussianBlur(sobelxy, (MASK_SIZE, MASK_SIZE), 0)
    gaussiany2 = cv.GaussianBlur(sobely2, (MASK_SIZE, MASK_SIZE), 0)

    det_M = gaussianx2 * gaussiany2 - gaussianxy * gaussianxy
    trace_M = gaussianx2 + gaussiany2

    H = det_M - K * trace_M * trace_M
    H = cv.normalize(H, None, alpha=0, beta=1, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)

    return H


def find_max(image, size, threshold):   # size - rozmiar maski filtra maksymalnego
    data_max = filters.maximum_filter(image, size)
    maxima = (image == data_max)
    diff = image > threshold
    maxima[diff == 0] = 0

    return np.nonzero(maxima)


def display_coords(coords, image, title, i):
    plt.subplot(1, 2, i)
    plt.imshow(image, cmap='gray')
    plt.plot(coords[1], coords[0], '*', color='r')
    plt.axis('off')
    plt.title(title)


h1 = calculate_H(f1, MASK_SIZE)
max_f1 = find_max(h1, MASK_SIZE, THRESH)

h2 = calculate_H(f2, MASK_SIZE)
max_f2 = find_max(h2, MASK_SIZE, THRESH)

display_coords(max_f1, f1, 'Fountain 1', 1)
display_coords(max_f2, f2, 'Fountain 2', 2)
plt.show()

print('Fountain - value h1:', h1)
print('Fountain - value h2:', h2)
print('Fountain - value max_f1:', max_f1)
print('Fountain - value max_f2:', max_f2)

h1 = calculate_H(b1, MASK_SIZE)
max_b1 = find_max(h1, MASK_SIZE, THRESH)

h2 = calculate_H(b2, MASK_SIZE)
max_b2 = find_max(h2, MASK_SIZE, THRESH)

display_coords(max_b1, b1, 'Building 1', 1)
display_coords(max_b2, b2, 'Building 2', 2)
plt.show()

print('Building - value b1:', b1)
print('Building - value b2:', b2)
print('Building - value max_b1:', max_b1)
print('Building - value max_b2:', max_b2)
