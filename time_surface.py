from re import L
from mpl_toolkits import mplot3d
import matplotlib.cm as cm

from math import ceil
import aedat
import numpy as np

import cv2
import os
import glob
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

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

# flattened pixel number
radii3 = [12, 13, 14, 20, 24, 28, 34, 37, 43, 46, 52, 56, 60, 66, 67, 68]
radii4 = [3, 4, 5, 11, 15, 19, 25, 27, 35, 36, 44, 45, 53, 55, 61, 65, 69, 75, 76, 77]

def medianFilter(rawIm):
    filtered = cv2.medianBlur(rawIm, 3) # Ksize aperture linear must be odd and greater than 1 ie 3, 5, 7... 
    
    return filtered

def read_aedat4(filename: str, time_step=66000, processAll=True, factor=1) -> np.ndarray:
    data = aedat.Decoder(filename)
    for packet in data:
        if processAll:
            total_events = len(packet['events'])
        else:
            total_events = time_step*factor
        frames = ceil(total_events/time_step)
        #image = np.zeros((frames, HEIGHT, WIDTH), dtype=np.ubyte)
        image = np.zeros((frames, HEIGHT, WIDTH, 2), dtype=np.uint32)

        frame = 0
        accumulated_time = 0
        intensity = 1
        for i in tqdm(range(0, total_events, dt), desc='processing events'):
            y = packet['events']['y'][i]
            x = packet['events']['x'][i]

            if frame == factor-1:
                pol = packet['events']['on'][i] # packet['events']['on'] = 1 means it's ON, packet['event']['on'] = 0 means it's OFF
                prev_state = image[frame][y][x][int(pol)] # This is for ArcStar
                prev_state_inv = image[frame][y][x][int(not pol)]
                image[frame][y][x][pol] = intensity

                # Check if it's a corner
                #if isCornerEFast(image[frame], x, y, int(pol)):
                #if isCornerArcStar(image[frame], prev_state, prev_state_inv, x, y, int(pol)):
                isCorner, output, _ = harrisCornerDet(image[frame], x, y, int(pol))
                if isCorner:
                    draw3DBarGraph(output, i, False)
                    #temp = image[frame][:, :, int(pol)].copy()
                    #cropped = crop(temp, x, y, 9) # 9 just like the Mueggler paper
                    #drawHeatMap(cropped, i, False)
                    #draw3DBarGraph(cropped, i, False)

            else:
                pol = packet['events']['on'][i] # packet['event']['on'] = 1 means it's ON, packet['event']['on'] = 0 means it's OFF
                image[frame][y][x][pol] = intensity

            if accumulated_time < time_step:
                accumulated_time += dt
                intensity += 1
            else:
                frame += 1
                accumulated_time = 0
                intensity = 1

    return image

def harrisCornerDet(image, centerX, centerY, pol, pixelSize=9):
    # neighbourhood size = 2
    # sobel operator aperture param = 8
    # k from det(M) - k(trace(M))^2 = 0.04

    max_scale = 1
    cs = max_scale * 4
    if (centerX < cs or centerX >= WIDTH-cs or centerY < cs or centerY >= HEIGHT-cs):
        return False, None, None

    isCorner = False
    check = crop(image[:, :, pol], centerX, centerY, pixelSize)
    norm = np.linalg.norm(check)
    check = ((check/norm) * 255).astype(np.uint8) # Normalise image
    output = check.copy()
    harris = cv2.cornerHarris(check, 2, 7, 0.04)
    dst = cv2.dilate(harris, None)
    threshold = np.argwhere(dst > 0.1*dst.max()) # Check if there are any strong corner cases
    if len(threshold) > 0:
        isCorner = True
    output[dst > 0.1*dst.max()] = [255]
    
    return isCorner, output, dst

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
    segment_new_min_t = image[centerY + SMALL_CIRCLE[0][1]][centerX + SMALL_CIRCLE[0][1]]
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

def drawHeatMap(image, number, display=True):
    fig = plt.figure()

    plt.imshow(image, cmap='coolwarm')
    m = cm.ScalarMappable(cmap=cm.coolwarm)
    #m.set_array(heatmap)
    m.set_array(image)
    plt.colorbar(m)

    plt.savefig("../Output/2D Heatmap No " + str(number) + ".jpg", dpi=300)
    
    if display:
        plt.show()
    plt.close()

def draw3DHeatMap(heatmap, number, display=True):
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2GRAY)

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    xx, yy = np.mgrid[0:heatmap.shape[0], 0:heatmap.shape[1]]
    ax.plot_surface(xx, yy, heatmap, rstride=1, cstride=1, cmap="coolwarm") #cmap="coolwarm" "gray"
    
    m = cm.ScalarMappable(cmap=cm.coolwarm)
    m.set_array(heatmap)
    plt.colorbar(m)
    #ax.view_init(0, 60) # parameters are elevation_angle in z plane, azimuth_angle in x,y plane

    plt.savefig("../Output/3DHeatmap No " + str(number) + ".jpg", dpi=300)

    if display:
        plt.show()
    plt.close()

def draw3DBarGraph(image, number, display=True):
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    xpos, ypos = np.meshgrid(np.arange(image.shape[1]), np.arange(image.shape[0]))
    xpos = xpos.flatten()
    ypos = ypos.flatten()
    zpos = image.flatten()

    # label the middle pixel, which is the largest value, with colour black, everything else is coloured blue
    colour = []
    for i in range(len(zpos)):
        if i == len(zpos) // 2:
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
    if not os.path.isdir("./Output"):
        os.mkdir("../Output")

    image = read_aedat4(aedat4_file, time_step=33000, processAll=False, factor=10)
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
