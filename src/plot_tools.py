from mpl_toolkits import mplot3d
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import cv2
import numpy as np
from  matplotlib.colors import LinearSegmentedColormap

# flattened pixel number
radii3 = [12, 13, 14, 20, 24, 28, 34, 37, 43, 46, 52, 56, 60, 66, 67, 68]
radii4 = [3, 4, 5, 11, 15, 19, 25, 27, 35, 36, 44, 45, 53, 55, 61, 65, 69, 75, 76, 77]

WIDTH = 240
HEIGHT = 180

def drawFeatureTrack2D(pastEventQueue, name, time_step):
    fig = plt.figure(figsize=(16,10))
    c = ["red","lightcoral","black", "palegreen","green"]
    v = [0,.4,.5,0.6,1.]
    l = list(zip(v,c))
    myCMap=LinearSegmentedColormap.from_list('rg',l, N=256)

    xData = []
    yData = []
    cData = []
    for i, (locX, locY, corner) in enumerate(pastEventQueue):
        xData.append(locX)
        yData.append(i)
        cData.append(corner)

    plt.scatter(xData, yData, s=1, c=cData, cmap=myCMap)
    plt.xlabel("width")
    plt.ylabel("event no")

    plt.savefig("../Output/2D Event Feature Track of " + str(time_step) + ".jpg", dpi=300)
    plt.show()
    plt.close()

def drawFeatureTrack3D(pastEventQueue, name, time_step):
    fig = plt.figure(figsize=(16,10))
    ax = plt.axes(projection='3d')
    ax.set_zscale("linear")
    c = ["red","lightcoral","black", "palegreen","green"]
    v = [0,.4,.5,0.6,1.]
    l = list(zip(v,c))
    myCMap=LinearSegmentedColormap.from_list('rg',l, N=256)

    xData = []
    yData = []
    zData = []
    cData = []
    for i, (locX, locY, corner) in enumerate(pastEventQueue):
        xData.append(locX)
        yData.append(locY)
        zData.append(i)
        cData.append(corner)

    ax.scatter3D(xData, yData, zData, s=1, c=cData, cmap=myCMap)
    ax.set_xlabel("width")
    ax.set_ylabel("height")
    ax.set_zlabel("event no")

    plt.savefig("../Output/3D Event Feature Track of " + str(time_step) + ".jpg", dpi=300)
    plt.show()
    plt.close()

def drawHeatMapSub(image, number, display=True, name=None, subplot=None, title=None):
    subplot.set_title(title, size=10)
    subplot.imshow(image, cmap='coolwarm')
    return subplot

def drawHeatMapWhole(image, number, display=True, name=None, subplot=None, title=None, symbol=None, cornerLocation=None):
    fig = plt.figure()

    plt.imshow(image, cmap='coolwarm')

    if symbol:
        row, col = [], []
        for i, j in cornerLocation:
            row.append(j)
            col.append(i)
        plt.scatter(row, col, color='r', marker=symbol)

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

def drawEventLocation(image, number, display=True, name=None, subplot=None, title=None, symbol=None, cornerLocation=None):
    fig = plt.figure()

    c = ["red","lightcoral","black", "palegreen","green"]
    #c = c[::-1]
    v = [0,.4,.5,0.6,1.]
    l = list(zip(v,c))
    myCMap=LinearSegmentedColormap.from_list('rg',l, N=256)

    row, col, value = [], [], []
    for (_, locX, locY, _, _) in cornerLocation:
        if image[locY][locX] != 128:
            row.append(locY)
            col.append(locX)
            value.append(image[locY][locX])
    plt.scatter(col, row, s=1, c=value, cmap=myCMap)
    plt.xlim(0, WIDTH)
    plt.ylim(0, HEIGHT)
    plt.gca().invert_yaxis()

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
    subplot.set_title(title, size=10)
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