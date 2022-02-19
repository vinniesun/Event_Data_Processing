from mpl_toolkits import mplot3d
import matplotlib.cm as cm

from math import ceil, pi, exp, sqrt
import aedat
import numpy as np

import cv2
import os
import glob
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

from scipy import signal, ndimage
from skimage.feature import corner_peaks
import copy

from collections import deque

aedat4_file = '../EBBINNOT_AEDAT4/Recording/20180711_Site1_3pm_12mm_01.aedat4'
image_path = '../images/corner/'
dt = 1
WIDTH = 240
HEIGHT = 180

# Circle Param
SMALL_CIRCLE = [[0, 3], [1, 3], [2, 2], [3, 1],
                [3, 0], [3, -1], [2, -2], [1, -3],
                [0, -3], [-1, -3], [-2, -2], [-3, -1],
                [-3, 0], [-3, 1], [-2, 2], [-1, 3]]
BIG_CIRCLE = [[0, 4], [1, 4], [2, 3], [3, 2],
              [4, 1], [4, 0], [4, -1], [3, -2],
              [2, -3], [1, -4], [0, -4], [-1, -4],
              [-2, -3], [-3, -2], [-4, -1], [-4, 0],
              [-4, 1], [-3, 2], [-2, 3], [-1, 4]]

# Sobel Operators
SOBEL_X = np.array([[-1, 0, 1],[-2, 0, 2],[-1, 0, 1]])
SOBEL_Y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

# flattened pixel number
radii3 = [12, 13, 14, 20, 24, 28, 34, 37, 43, 46, 52, 56, 60, 66, 67, 68]
radii4 = [3, 4, 5, 11, 15, 19, 25, 27, 35, 36, 44, 45, 53, 55, 61, 65, 69, 75, 76, 77]

def medianFilter(rawIm):
    filtered = cv2.medianBlur(rawIm, 3) # Ksize aperture linear must be odd and greater than 1 ie 3, 5, 7... 
    
    return filtered

def read_aedat4(filename: str, time_step=66000, processAll=True, factor=1, eventSpecified=False, eventNo=None) -> np.ndarray:
    data = aedat.Decoder(filename)
    for packet in data:
        #if processAll:
        #    total_events = len(packet['events'])
        #else:
        #    total_events = time_step*factor
        #frames = ceil(total_events/time_step)
        #image = np.zeros((frames, HEIGHT, WIDTH), dtype=np.ubyte)
        total_events = len(packet['events'])
        frames = ceil(packet['events']['t'][-1]/time_step)
        image = np.zeros((frames, HEIGHT, WIDTH, 2), dtype=np.uint32)

        frame = 0
        accumulated_time = 0
        last_event = 0
        
        for i in tqdm(range(0, total_events, dt), desc='processing events'):
            y = packet['events']['y'][i]
            x = packet['events']['x'][i]
            pol = packet['events']['on'][i] # packet['events']['on'] = 1 means it's ON, packet['event']['on'] = 0 means it's OFF

            prev_state = image[frame][y][x][int(pol)] # This is for ArcStar
            prev_state_inv = image[frame][y][x][int(not pol)] # This is for ArcStar

            image[frame][y][x][int(pol)] = packet['events']['t'][i]
            image[frame][:, :, int(pol)] = image[frame][:, :, int(pol)] - last_event # Clip all values that are negative to zero
            image[frame][image[frame] < 0] = 0

            if eventSpecified and i == eventNo:
            #if frame == factor-1 and i%time_step == eventNo:
                drawHeatMapWhole(image[frame][:, :, int(pol)], i, eventSpecified, name="Full SAE" + str(time_step) + "us")

                noCorner, output, cornerLocation = openCVHarrisCornerDet(image[frame], x, y, int(pol))
                #noCorner, output, cornerLocation = customHarrisCornerDet(image[frame], x, y, int(pol))
                drawHeatMapWhole(output, i, eventSpecified, name="All Corners " + str(time_step) + "us")
                noRow, noCol = int((sqrt(noCorner))), ceil(noCorner / int(sqrt(noCorner)))

                # Plot Harris Response
                fig, ax = plt.subplots(noRow, noCol,figsize=(12, 9))
                plt.tight_layout()
                for j, (row, col) in enumerate(cornerLocation):
                    temp = crop(image[frame][:, :, int(pol)], col, row, 9)
                    ax[j // noCol][j % noCol] = drawHeatMapSub(temp, i, subplot=ax[j // noCol][j % noCol], title="Corner No. " + str(j))
                plt.savefig("../Output/" + str(time_step) + "us 2D Harris Corner of Event " + str(eventNo) + ".jpg", dpi=300)
                plt.show()
                plt.close(fig)

                fig = plt.figure(figsize=(12, 9))
                plt.tight_layout()
                for j, (row, col) in enumerate(cornerLocation):
                    ax = fig.add_subplot(noRow, noCol, j+1, projection='3d')
                    temp = crop(image[frame][:, :, int(pol)], col, row, 9)
                    ax = draw3DBarGraphSub(temp, i, subplot=ax, title="Corner No. " + str(j))
                plt.savefig("../Output/" + str(time_step) + "us 3D Barplot of Harris Corner of Event " + str(eventNo) + ".jpg", dpi=300)
                plt.show()
                plt.close(fig)
                
                # Plot eFast Response
                fig, ax = plt.subplots(noRow, noCol,figsize=(12, 9))
                plt.tight_layout()
                for j, (row, col) in enumerate(cornerLocation):
                    if isCornerEFast(image[frame], col, row, int(pol)):
                        temp = crop(image[frame][:, :, int(pol)], col, row, 9)
                        ax[j // noCol][j % noCol] = drawHeatMapSub(temp, i, subplot=ax[j // noCol][j % noCol], title="Corner No. " + str(j))
                plt.savefig("../Output/" + str(time_step) + "us 2D eFast Corner of Event " + str(eventNo) + ".jpg", dpi=300)
                plt.show()
                plt.close(fig)

                fig = plt.figure(figsize=(12, 9))
                plt.tight_layout()
                for j, (row, col) in enumerate(cornerLocation):
                    ax = fig.add_subplot(noRow, noCol, j+1, projection='3d')
                    if isCornerEFast(image[frame], col, row, int(pol)):
                        temp = crop(image[frame][:, :, int(pol)], col, row, 9)
                        ax = draw3DBarGraphSub(temp, i, subplot=ax, title="Corner No. " + str(j))
                plt.savefig("../Output/" + str(time_step) + "us 3D Barplot of eFast Corner of Event " + str(eventNo) + ".jpg", dpi=300)
                plt.show()
                plt.close(fig)

                # Plot ArcStar Response
                fig, ax = plt.subplots(noRow, noCol,figsize=(12, 9))
                plt.tight_layout()
                for j, (row, col) in enumerate(cornerLocation):
                    if isCornerArcStar(image[frame], prev_state, prev_state_inv, col, row, int(pol)):
                        temp = crop(image[frame][:, :, int(pol)], col, row, 9)
                        ax[j // noCol][j % noCol] = drawHeatMapSub(temp, i, subplot=ax[j // noCol][j % noCol], title="Corner No. " + str(j))
                plt.savefig("../Output/" + str(time_step) + "us 2D ArcStar Corner of Event " + str(eventNo) + ".jpg", dpi=300)
                plt.show()
                plt.close(fig)

                fig = plt.figure(figsize=(12, 9))
                plt.tight_layout()
                for j, (row, col) in enumerate(cornerLocation):
                    ax = fig.add_subplot(noRow, noCol, j+1, projection='3d')
                    if isCornerArcStar(image[frame], prev_state, prev_state_inv, col, row, int(pol)):
                        temp = crop(image[frame][:, :, int(pol)], col, row, 9)
                        ax = draw3DBarGraphSub(temp, i, subplot=ax, title="Corner No. " + str(j))
                plt.savefig("../Output/" + str(time_step) + "us 3D Barplot of ArcStar Corner of Event " + str(eventNo) + ".jpg", dpi=300)
                plt.show()
                plt.close(fig)

                break

            if i == 0:
                accumulated_time = 0
            else:
                accumulated_time += (packet['events']['t'][i] - packet['events']['t'][i-1])
            
            if accumulated_time >= time_step:
                frame += 1
                accumulated_time = 0
                image[frame] = image[frame-1]
            
            last_event = packet['events']['t'][i]

    return image

def normalise(image):
    oldMin = 0
    oldMax = image.max()
    newMax = 255

    normalised = (image - oldMin)/(oldMax - oldMin)
    normalised *= newMax
    normalised = normalised.astype(np.uint8)

    return normalised

class eHarris():
    def __init__(self, pixelSize=9, queueSize=25, harrisThreshold=8.0, windowSize=4, kernelSize=5):
        self.queue = deque(maxlen=25)
        self.harrisThreshold = harrisThreshold
        self.windowSize = windowSize
        self.kernelSize = kernelSize
        self.last_score = None
        self.Sx = np.empty(kernelSize)
        self.Dx = np.empty(kernelSize)

        for i in range(kernelSize):
            self.Sx[i] = self.factorial(kernelSize-1)/(self.factorial(kernelSize-1-i)*self.factorial(i))
            self.Dx[i] = self.pasc(i, kernelSize-2) - self.pasc(i-1, kernelSize-2)
        
        self.Gx = self.Sx * np.transpose(self.Dx)
        self.Gx = self.Gx / np.max(self.Gx)

        self.sigma = 1
        self.A = 1.0/(2.0*pi*self.sigma**2)
        self.l2 = (2*windowSize + 2 - kernelSize)/2
        self.h = np.array((2*self.l2+1, 2*self.l2+1))

        for row in range(-1*self.l2, self.l2+1):
            for col in range(-1*self.l2, self.l2+1):
                h_xy = self.A * exp(-(row**2 + col**2)/(2*self.sigma**2))
                self.h[self.l2+row][self.l2+col] = h_xy

        self.h /= np.sum(self.h)

    def getLastScore(self):
        return self.last_score

    def factorial(self, n):
        if n>1:
            return n*self.factorial(n-1)
        else:
            return 1

    def pasc(self, k, n):
        if k>=0 and k<=n:
            return self.factorial(n)/(self.factorial(n-k)*self.factorial(k))
        else:
            return 0

    def getHarrisScore(self, image, centerX, centerY, pol):
        # Check if it's at the border
        if (centerX < self.windowSize or centerX > WIDTH - self.windowSize or centerY < self.windowSize or centerY > HEIGHT - self.windowSize):
            return self.harrisThreshold - 10.0

        local_frame = crop(image, centerX, centerY, self.windowSize)

        l = 2*self.windowSize + 2 - self.kernelSize
        dx = np.zeros((l, l))
        dy = np.zeros((l, l))

        for row in range(l):
            for col in range(l):
                for krow in range(self.kernelSize):
                    for kcol in range(self.kernelSize):
                        dx[row][col] += local_frame[row+krow][col+kcol] * self.Gx[krow][kcol]
                        dx[row][col] += local_frame[row+krow][col+kcol] * self.Gx[kcol][krow]

        a, b, d = 0.0, 0.0, 0.0
        for row in range(l):
            for col in range(l):
                a += self.h[row][col] * dx[row][col] * dx[row][col]
                b += self.h[row][col] * dx[row][col] * dy[row][col]
                d += self.h[row][col] * dy[row][col] * dy[row][col]

        score = a*d - b**2 - 0.04*(a+d)**2

        return score

    def isFeature(self, image, centerX, centerY, pol):
        score = self.harrisThreshold - 10.0

        if self.queue[-1]: # the queue is full
            self.queue.popleft()
            self.queue.append((centerX, centerY, pol))
            score = self.getHarrisScore()
            self.last_score = score
        else:
            self.queue.append((centerX, centerY, pol))
            if self.queue[-1]:
                score = self.getHarrisScore(image[:,:,pol], centerX, centerY, pol)
                self.last_score = score

        return score > self.harrisThreshold

def gradient(image, filter, mode="same"):
    return signal.convolve2d(image, filter, mode)

def customHarrisCornerDet(image, centerX, centerY, pol, pixelSize=9, k=0.04):
    # Check if we are at the border
    max_scale = 1
    cs = max_scale * 4
    if (centerX < cs or centerX >= WIDTH-cs or centerY < cs or centerY >= HEIGHT-cs):
        return False, None

    check = image[:, :, pol]
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

    """
    isCorner = False
    cropped = crop(image[:, :, pol], centerX, centerY, pixelSize)

    grad_x = gradient(cropped, SOBEL_X)
    grad_y = gradient(cropped, SOBEL_Y)

    Ixx = ndimage.gaussian_filter(grad_x**2, sigma=1)
    Iyy = ndimage.gaussian_filter(grad_y**2, sigma=1)
    Ixy = ndimage.gaussian_filter(grad_x*grad_y, sigma=1)

    detA = Ixx * Iyy - Ixy**2
    traceA = Ixx + Iyy
    harris = detA - k*traceA**2

    output = cropped.copy()
    
    #for row, response in enumerate(harris):
    #    for col, r in enumerate(response):
    #        if r > 0:
    #            isCorner = True
    #            output[row][col] = cropped[row][col]
    #        else:
    #            output[row][col] = cropped[row][col]
    
    if harris[pixelSize // 2][pixelSize // 2] > 0:
        isCorner = True

    return isCorner, output
    """

def openCVHarrisCornerDet(image, centerX, centerY, pol, pixelSize=9, k=0.04):
    # neighbourhood size = 2
    # sobel operator aperture param = 8
    # k from det(M) - k(trace(M))^2 = 0.04

    max_scale = 1
    cs = max_scale * 4
    if (centerX < cs or centerX >= WIDTH-cs or centerY < cs or centerY >= HEIGHT-cs):
        return False, None, None

    check = image[:, :, pol]
    noOfCorners, output, cornerLocation = None, np.zeros((check.shape[0], check.shape[1])), [None]
    #norm_image = check * (255/image.max())
    #norm_image = norm_image.astype(np.uint8)
    norm_image = normalise(check)
    harris = cv2.cornerHarris(norm_image, 2, 7, k)
    dst = cv2.dilate(harris, None)

    noOfCorners = np.count_nonzero(dst > 0.99*dst.max())
    cornerLocation = np.argwhere(dst > 0.99*dst.max()) # output is row*col of index
    output[dst > 0.4*dst.max()] = [255]

    # output to show:
    #   no of corners (for subplot division), output corner image, list of corner locations, 
    return noOfCorners, output, cornerLocation

    """
    isCorner = False
    check = crop(image[:, :, pol], centerX, centerY, pixelSize)
    
    output = copy.deepcopy(check)
    norm = np.linalg.norm(check)
    check = ((check/norm) * 255).astype(np.uint8) # Normalise image
    harris = cv2.cornerHarris(check, 2, 7, k)
    dst = cv2.dilate(harris, None)
    #threshold = np.argwhere(dst > 0.1*dst.max()) # Check if there are any strong corner cases
    #if len(threshold) > 0:
    #    isCorner = True
    #output[dst > 0.1*dst.max()] = [255]
    threshold = 0.1*dst.max()
    if dst[pixelSize // 2][pixelSize // 2] >= threshold:
        isCorner = True

    return isCorner, output, dst
    """

# Size can be either 5 or 10
def crop(image, startX, startY, size):
    half = size // 2
    return image[startY-half:startY+(size-half), startX-half:startX+(size-half)]

def getMaxIndex(image):
    maxIndices = np.where(image == np.amax(image))
    result = list(zip(maxIndices[0], maxIndices[1]))

    return result

def isCornerEFast(img, centerX, centerY, pol):
    found = False
    smallCount, bigCount = 0, 0
    image = img[:, :, pol]

    # Check if it's too close to the border of the SAE
    max_scale = 1
    cs = max_scale * 4
    if (centerX < cs or centerX >= WIDTH-cs or centerY < cs or centerY >= HEIGHT-cs):
        return False
    
    for i in range(len(SMALL_CIRCLE)):
        for streak_size in range(3, 7):
            if image[centerY + SMALL_CIRCLE[i][1]][centerX + SMALL_CIRCLE[i][0]] < image[centerY + SMALL_CIRCLE[(i-1+16)%16][1]][centerX + SMALL_CIRCLE[(i-1+16)%16][0]]:
                continue

            if image[centerY + SMALL_CIRCLE[(i + streak_size - 1)%16][1]][centerX + SMALL_CIRCLE[(i + streak_size - 1)%16][0]] < image[centerY + SMALL_CIRCLE[(i+streak_size)%16][1]][centerX + SMALL_CIRCLE[(i+streak_size)%16][0]]:
                continue

            min_t = image[centerY + SMALL_CIRCLE[i][1]][centerX + SMALL_CIRCLE[i][0]]

            for j in range(1, streak_size):
                tj = image[centerY + SMALL_CIRCLE[(i+j)%16][1]][centerX + SMALL_CIRCLE[(i+j)%16][0]]
                if tj < min_t:
                    min_t = tj
            
            did_break = False
            for j in range(streak_size, len(SMALL_CIRCLE)):
                tj = image[centerY + SMALL_CIRCLE[(i+j)%16][1]][centerX + SMALL_CIRCLE[(i+j)%16][0]]
                if tj >= min_t:
                    did_break = True
                    break

            if not did_break:
                found = True
                break

        if found:
            break

    if found:
        found = False
        for i in range(len(BIG_CIRCLE)):
            for streak_size in range(4, 9):
                if image[centerY + BIG_CIRCLE[i][1]][centerX + BIG_CIRCLE[i][0]] < image[centerY + BIG_CIRCLE[(i-1+20)%20][1]][centerX + BIG_CIRCLE[(i-1+20)%20][0]]:
                    continue

                if image[centerY + BIG_CIRCLE[(i + streak_size - 1)%20][1]][centerX + BIG_CIRCLE[(i + streak_size - 1)%20][0]] < image[centerY + BIG_CIRCLE[(i+streak_size)%20][1]][centerX + BIG_CIRCLE[(i+streak_size)%20][0]]:
                    continue

                min_t = image[centerY + BIG_CIRCLE[i][1]][centerX + BIG_CIRCLE[i][0]]
                for j in range(1, streak_size):
                    tj = image[centerY + BIG_CIRCLE[(i+j)%20][1]][centerX + BIG_CIRCLE[(i+j)%20][0]]
                    if tj < min_t:
                        min_t = tj
                
                did_break = False
                for j in range(streak_size, len(BIG_CIRCLE)):
                    tj = image[centerY + BIG_CIRCLE[(i+j)%20][1]][centerX + BIG_CIRCLE[(i+j)%20][0]]
                    if tj >= min_t:
                        did_break = True
                        break

                if not did_break:
                    found = True
                    break

            if found:
                break

    return found

def isCornerArcStar(img, prev_state, prev_state_inv, centerX, centerY, pol, filter_threshold=0.05):
    if pol == 0:
        pol_inv = 1
    else:
        pol_inv = 0

    t_last = prev_state
    t_last_inv = prev_state_inv

    # Filter out redundant spikes, e.g. spikes of the same polarity that's fired off consecutively in short period
    if ((img[centerY][centerX][pol] > t_last + filter_threshold) or (t_last_inv > t_last)):
        t_last = img[centerY][centerX][pol]
    else:
        t_last = img[centerY][centerX][pol]
        return False

    # Check if it's too close to the border of the SAE
    max_scale = 1
    cs = max_scale * 4
    if (centerX < cs or centerX >= WIDTH-cs or centerY < cs or centerY >= HEIGHT-cs):
        return False

    found = False
    image = img[:, :, pol]
    segment_new_min_t = image[centerY + SMALL_CIRCLE[0][1]][centerX + SMALL_CIRCLE[0][0]]
    arc_left_idx, arc_right_idx = 0, 0 # this is the CCW & CW index in the original paper
    
    for i in range(1, len(SMALL_CIRCLE)):
        t = image[centerY + SMALL_CIRCLE[i][1]][centerX + SMALL_CIRCLE[i][0]]
        if t > segment_new_min_t:
            segment_new_min_t = t
            arc_right_idx = i
    
    arc_left_idx = (arc_right_idx - 1 + len(SMALL_CIRCLE))%len(SMALL_CIRCLE)
    arc_right_idx = (arc_right_idx + 1)%len(SMALL_CIRCLE)

    arc_left_val = image[centerY + SMALL_CIRCLE[arc_left_idx][1]][centerX + SMALL_CIRCLE[arc_left_idx][0]]
    arc_right_val = image[centerY + SMALL_CIRCLE[arc_right_idx][1]][centerX + SMALL_CIRCLE[arc_right_idx][0]]
    arc_left_min_t = arc_left_val
    arc_right_min_t = arc_right_val

    for j in range(0, 3): # 3 is the smallest segment length of an acceptable arc
        if arc_right_val > arc_left_val:
            if arc_right_min_t < segment_new_min_t:
                segment_new_min_t = arc_right_min_t
            
            arc_right_idx = (arc_right_idx + 1)%len(SMALL_CIRCLE)
            arc_right_val = image[centerY + SMALL_CIRCLE[arc_right_idx][1]][centerX + SMALL_CIRCLE[arc_right_idx][0]]
            if arc_right_val < arc_right_min_t:
                arc_right_min_t = arc_right_val

        else:
            if arc_left_min_t < segment_new_min_t:
                segment_new_min_t = arc_left_min_t

            arc_left_idx = (arc_left_idx - 1 + len(SMALL_CIRCLE))%len(SMALL_CIRCLE)
            arc_left_val = image[centerY + SMALL_CIRCLE[arc_left_idx][1]][centerX + SMALL_CIRCLE[arc_left_idx][0]]
            if arc_left_val < arc_left_min_t:
                arc_left_min_t = arc_left_val
    
    newest_segment_size = 3

    for j in range(3, len(SMALL_CIRCLE)): # look through the rest of the circle
        if arc_right_val > arc_left_val:
            if arc_right_val >= segment_new_min_t:
                newest_segment_size = j+1
                if arc_right_min_t < segment_new_min_t:
                    segment_new_min_t = arc_right_min_t
        
            arc_right_idx = (arc_right_idx+1)%len(SMALL_CIRCLE)
            arc_right_val = image[centerY + SMALL_CIRCLE[arc_right_idx][1]][centerX + SMALL_CIRCLE[arc_right_idx][0]]
            if arc_right_val < arc_right_min_t:
                arc_right_min_t = arc_right_val

        else:
            if arc_left_val >= segment_new_min_t:
                newest_segment_size = j+1
                if arc_left_min_t < segment_new_min_t:
                    segment_new_min_t = arc_left_min_t

            arc_left_idx = (arc_left_idx - 1 + len(SMALL_CIRCLE))%len(SMALL_CIRCLE)
            arc_left_val = image[centerY + SMALL_CIRCLE[arc_left_idx][1]][centerX + SMALL_CIRCLE[arc_left_idx][0]]
            if arc_left_val < arc_left_min_t:
                arc_left_min_t = arc_left_val

    if ((newest_segment_size <= 6) or (newest_segment_size >= len(SMALL_CIRCLE) - 6) and (newest_segment_size <= (len(SMALL_CIRCLE) - 3))): # Check the arc size satisfy the requirement
        found = True
    
    # Search through the large circle if small circle verifies
    if found:
        found = False
        segment_new_min_t = image[centerY + BIG_CIRCLE[0][1]][centerX + BIG_CIRCLE[0][0]]
        arc_right_idx = 0

        for i in range(1, len(BIG_CIRCLE)):
            t = image[centerY + BIG_CIRCLE[i][1]][centerX + BIG_CIRCLE[i][0]]
            if t > segment_new_min_t:
                segment_new_min_t = t
                arc_right_idx = i

        arc_left_idx = (arc_right_idx - 1 + len(BIG_CIRCLE))%len(BIG_CIRCLE)
        arc_right_idx = (arc_right_idx + 1)%len(BIG_CIRCLE)
        arc_left_val = image[centerY + BIG_CIRCLE[arc_left_idx][1]][centerX + BIG_CIRCLE[arc_left_idx][0]]
        arc_right_val = image[centerY + BIG_CIRCLE[arc_right_idx][1]][centerX + BIG_CIRCLE[arc_right_idx][0]]

        arc_left_min_t = arc_left_val
        arc_right_min_t = arc_right_val

        for j in range(1, 4):
            if (arc_right_val > arc_left_val):
                if (arc_right_min_t > arc_left_min_t):
                    segment_new_min_t
                arc_right_idx = (arc_right_idx+1)%len(BIG_CIRCLE)
                arc_right_val = image[centerY + BIG_CIRCLE[arc_right_idx][1]][centerX + BIG_CIRCLE[arc_right_idx][0]]

                if arc_right_val < arc_right_min_t:
                    arc_right_min_t = arc_right_val
            else:
                if arc_left_min_t < segment_new_min_t:
                    segment_new_min_t = arc_left_min_t
                arc_left_idx = (arc_left_idx - 1 + len(BIG_CIRCLE))%len(BIG_CIRCLE)
                arc_left_val = image[centerY + BIG_CIRCLE[arc_left_idx][1]][centerX + BIG_CIRCLE[arc_left_idx][0]]
                if arc_left_val < arc_left_min_t:
                    arc_left_min_t = arc_left_val

        newest_segment_size = 4

        for j in range(4, 8):
            if arc_right_val > arc_left_val:
                if arc_right_val >= segment_new_min_t:
                    newest_segment_size = j+1
                    if arc_right_min_t < segment_new_min_t:
                        segment_new_min_t = arc_right_min_t

                arc_right_idx = (arc_right_idx + 1)%len(BIG_CIRCLE)
                arc_right_val = image[centerY + BIG_CIRCLE[arc_right_idx][1]][centerX + BIG_CIRCLE[arc_right_idx][0]]
                if arc_right_val < arc_right_min_t:
                    arc_right_min_t = arc_right_val
            
            else:
                if arc_left_val >= segment_new_min_t:
                    newest_segment_size = j+1
                    if arc_left_min_t < segment_new_min_t:
                        segment_new_min_t = arc_left_min_t
                    
                arc_left_idx = (arc_left_idx - 1 + len(BIG_CIRCLE))%len(BIG_CIRCLE)
                arc_left_val = image[centerY + BIG_CIRCLE[arc_left_idx][1]][centerX + BIG_CIRCLE[arc_left_idx][0]]
                if arc_left_val < arc_left_min_t:
                    arc_left_min_t = arc_left_val

            if ((newest_segment_size <= 8) or ((newest_segment_size >= (len(BIG_CIRCLE) - 8)) and (newest_segment_size <= (len(BIG_CIRCLE) - 4)))):
                found = True

    return found

def drawHeatMapSub(image, number, display=True, name=None, subplot=None, title=None):
    subplot.set_title(title)
    subplot.imshow(image, cmap='coolwarm')
    return subplot

def drawHeatMapWhole(image, number, display=True, name=None, subplot=None, title=None):
    fig = plt.figure()

    plt.imshow(image, cmap='coolwarm')
    m = cm.ScalarMappable(cmap=cm.coolwarm)
    m.set_array(image)
    plt.colorbar(m)

    if name:
        plt.savefig("../Output/2D Heatmap No " + str(number) + " " + name + ".jpg", dpi=300)
    else:
        plt.savefig("../Output/2D Heatmap No " + str(number) + ".jpg", dpi=300)
    
    if display:
        plt.show()
    plt.close()

def draw3DHeatMap(heatmap, number, display=True, name=None):
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2GRAY)

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    xx, yy = np.mgrid[0:heatmap.shape[0], 0:heatmap.shape[1]]
    ax.plot_surface(xx, yy, heatmap, rstride=1, cstride=1, cmap="coolwarm") #cmap="coolwarm" "gray"
    
    #m = cm.ScalarMappable(cmap=cm.coolwarm)
    #m.set_array(heatmap)
    #plt.colorbar(m)
    #ax.view_init(0, 60) # parameters are elevation_angle in z plane, azimuth_angle in x,y plane

    if name:
        plt.savefig("../Output/3D Heatmap No " + str(number) + " " + name + ".jpg", dpi=300)
    else:
        plt.savefig("../Output/3D Heatmap No " + str(number) + ".jpg", dpi=300)

    if display:
        plt.show()
    plt.close()

def draw3DBarGraphSub(image, number, display=True, subplot=None, title=None):
    subplot.set_title(title)
    xpos, ypos = np.meshgrid(np.arange(image.shape[1]), np.arange(image.shape[0]))
    xpos = xpos.flatten()
    ypos = ypos.flatten()
    zpos = image.flatten()

    # label the middle pixel, which is the largest value, with colour black, everything else is coloured blue
    colour = []
    for i in range(len(zpos)):
        #if i == len(zpos) // 2:
        if i == 40:
            colour.append('black')
        elif i in radii3:
            colour.append('red')
        elif i in radii4:
            colour.append('green')
        else:
            colour.append('blue')

    subplot.bar3d(xpos, ypos, np.zeros(len(zpos)), 1, 1, zpos, color=colour)

    return subplot

def draw3DBarGraphWhole(image, number, display=True):
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    xpos, ypos = np.meshgrid(np.arange(image.shape[1]), np.arange(image.shape[0]))
    xpos = xpos.flatten()
    ypos = ypos.flatten()
    zpos = image.flatten()

    # label the middle pixel, which is the largest value, with colour black, everything else is coloured blue
    colour = []
    for i in range(len(zpos)):
        #if i == len(zpos) // 2:
        if i == 40:
            colour.append('black')
        elif i in radii3:
            colour.append('red')
        elif i in radii4:
            colour.append('green')
        else:
            colour.append('blue')

    ax.bar3d(xpos, ypos, np.zeros(len(zpos)), 1, 1, zpos, color=colour) #xpos, ypos, np.zeros(len(zpos)) gives the position of each bar. 1, 1, zpos give the width, depth and height of the bars
    
    plt.savefig("../Output/3DBar Graph No " + str(number) + ".jpg", dpi=300)

    if display:
        plt.show()
    plt.close()

if __name__ == "__main__":
    if not os.path.isdir("../Output"):
        os.mkdir("../Output")

    image = read_aedat4(aedat4_file, time_step=33000, processAll=False, eventSpecified=True, eventNo=6000)
    """
    image = image[-1]
    filtered = medianFilter(image) # apply median filter to image

    # With Filtering
    corner_filtered_output, filtered_dst = harrisCornerDet(filtered) # Get the Harris Corner Detected
    heatmap_filtered = cv2.applyColorMap(corner_filtered_output, cv2.COLORMAP_AUTUMN) # Get the Heatmap of the Corner Detected

    # Without Filtering
    corner_output, dst = harrisCornerDet(image)
    heatmap = cv2.applyColorMap(corner_output, cv2.COLORMAP_AUTUMN)

    # Get the index for the largest value i.e. latest event
    maxIndices = getMaxIndex(filtered)
    for i, (row, col) in tqdm(enumerate(maxIndices), desc="generate crops"):
        cropped = crop(image, col, row, 9) # 9 just like the Mueggler paper
        # Draw the result
        if isCorner(cropped):
            drawHeatMap(cropped, i, False)
            draw3DBarGraph(cropped, i, False)
    """
