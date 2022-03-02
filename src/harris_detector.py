from scipy import signal, ndimage
from skimage.feature import corner_peaks
import numpy as np
import cv2

from .util import normalise

# The HEIGHT and WIDTH of a DAVIS Camera
WIDTH = 240
HEIGHT = 180

# Sobel Operators
SOBEL_X = np.array([[-1, 0, 1],[-2, 0, 2],[-1, 0, 1]])
SOBEL_Y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

def gradient(image, filter, mode="same"):
    return signal.convolve2d(image, filter, mode)

def customHarrisCornerDet(image, centerX, centerY, pol, pixelSize=9, k=0.04):
    # Check if we are at the border
    max_scale = 1
    cs = max_scale * 4
    if (centerX < cs or centerX >= WIDTH-cs or centerY < cs or centerY >= HEIGHT-cs):
        return False, None

    check = image
    noOfCorners, output, cornerLocation = None, np.zeros((check.shape[0], check.shape[1])), [None]

    grad_x = gradient(check, SOBEL_X)
    grad_y = gradient(check, SOBEL_Y)

    Ixx = ndimage.gaussian_filter(grad_x**2, sigma=1)
    Iyy = ndimage.gaussian_filter(grad_y**2, sigma=1)
    Ixy = ndimage.gaussian_filter(grad_x*grad_y, sigma=1)

    detA = Ixx * Iyy - Ixy**2
    traceA = Ixx + Iyy
    harris = detA - k*traceA**2

    cornerLocation = corner_peaks(harris) # Shape is row*col
    noOfCorners = len(cornerLocation)

    for row, col in cornerLocation:
        output[row][col] = 255

    return noOfCorners, output, cornerLocation

def openCVHarrisCornerDet(image, centerX, centerY, pol, pixelSize=9, k=0.04):
    # neighbourhood size = 2
    # sobel operator aperture param = 8
    # k from det(M) - k(trace(M))^2 = 0.04

    max_scale = 1
    cs = max_scale * 4
    if (centerX < cs or centerX >= WIDTH-cs or centerY < cs or centerY >= HEIGHT-cs):
        return False, None, None

    check = image
    noOfCorners, output, cornerLocation = None, np.zeros((check.shape[0], check.shape[1])), [None]
    #norm_image = check * (255/image.max())
    #norm_image = norm_image.astype(np.uint8)
    norm_image = normalise(check)
    harris = cv2.cornerHarris(norm_image, 2, 7, k)
    dst = cv2.dilate(harris, None)

    noOfCorners = np.count_nonzero(dst > 0.6*dst.max())
    cornerLocation = np.argwhere(dst > 0.6*dst.max()) # output is row*col of index
    output[dst > 0.6*dst.max()] = [255]

    # output to show:
    #   no of corners (for subplot division), output corner image, list of corner locations, 
    return noOfCorners, output, cornerLocation