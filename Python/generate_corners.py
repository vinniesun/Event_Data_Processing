from tqdm import tqdm
import numpy as np
import os
import matplotlib

from src.util import *
from src.plot_tools import *
from src.process_events import *

ACCUMULATE_TIME = 66000*4

def update_time_surface_new(time_surface: np.array, t, x, y, p, time_threshold: int=66000*4, prev_time=0, bits=1, mode: str="absolute"):
    try:
        if mode == "absolute":
            time_surface[y][x][int(p)] = t // 2**16
        elif mode == "delta":
            deltaT = ((t - prev_time)//4)*4

            time_surface[:, :, int(p)] -= deltaT
            time_surface[y][x][int(p)] = time_threshold
            mask = np.where(time_surface < 0)

            time_surface[mask] = 0
        elif mode == "bits":
            maxVal = 2 ** bits
            deltaT = t - prev_time

            time_surface[:, :, int(p)] -= deltaT
            time_surface[y][x][int(p)] = maxVal
            
            mask = np.where(time_surface < 0)
            time_surface[mask] = 0

        return time_surface

    except ValueError:
        print("Wrong Mode. Got " + mode + " as the mode, expected absolute, delta or bits")

def drawFeatureTrack3D_New(pastEventQueue, name, time_step):
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
    for i, (t, locX, locY, corner) in enumerate(pastEventQueue):
        xData.append(t)
        yData.append(locX)
        zData.append(locY)
        cData.append(corner)

    ax.scatter3D(xData, yData, zData, s=1, c=cData, cmap=myCMap)
    ax.set_xlabel("time")
    ax.set_ylabel("width")
    ax.set_zlabel("height")

    plt.savefig("../Output/3D Event Feature Track of " + name + str(time_step) + ".jpg", dpi=300)
    #plt.show()
    plt.close()

def extract_events(filename):
    length = 0
    with open(filename, 'r') as f:
        for _ in f:
            length += 1

    # Extract events
    events = [None] * length
    with open(filename, 'r') as f:
        for i, line in enumerate(tqdm(f)):
            event = line.split(",")
            assert len(event) == 4, "the line should contain only four elements: t, x, y, p"
            events[i] = (int(event[0]), int(event[1]), int(event[2]), int(event[3]))
    
    return events

def extract_events_new(filename):
    length = 0
    with open(filename, 'r') as f:
        for _ in f:
            length += 1

    # Extract events
    events = [None] * length
    with open(filename, 'r') as f:
        for i, line in enumerate(tqdm(f)):
            event = line.split(",")
            assert len(event) == 5, "the line should contain only four elements: t, x, y, p"
            events[i] = (int(event[0]), int(event[2]), int(event[3]), int(event[4]))
    
    return events

if __name__ == "__main__":
    all_events = extract_events("./shapes_translation_absolute/All_events.txt")
    efast_events_abs = extract_events("./shapes_translation_absolute/eFast_Corners.txt")
    arcstar_events_abs = extract_events("./shapes_translation_absolute/arcStar_Corners.txt")

    efast_events_factor = extract_events("./shapes_translation_factor/eFast_Corners.txt")
    arcstar_events_factor = extract_events("./shapes_translation_factor/arcStar_Corners.txt")
    

    if not os.path.isdir("./Output/eFast/"):
        os.makedirs("./Output/eFast/")
    if not os.path.isdir("./Output/ArcStar/"):
        os.makedirs("./Output/ArcStar/")
    if not os.path.isdir("../Output/"):
        os.makedirs("../Output/")

    #drawFeatureTrack3D_New(arcstar_events, "Poster Translation Arcstar Feature Track", 1)
    #drawFeatureTrack3D_New(efast_events, "Poster Translation eFast Feature Track", 1)
    #drawFeatureTrack3D_New(arcstar_events_33ms, "Shapes Rotation ArcStar Feature Track", 33000)
    #drawFeatureTrack3D_New(efast_events_33ms, "Shapes Rotation eFast Feature Track", 33000)
    #drawFeatureTrack3D_New(arcstar_events_66ms, "Shapes Rotation ArcStar Feature Track", 66000)
    #drawFeatureTrack3D_New(efast_events_66ms, "Shapes Rotation eFast Feature Track", 66000)
    #drawFeatureTrack3D_New(arcstar_events_664, "Shapes Rotation ArcStar Feature Track", 66000*4)
    #drawFeatureTrack3D_New(efast_events_664, "Shapes Rotation eFast Feature Track", 66000*4)
    #drawFeatureTrack3D_New(arcstar_events_factor, "Poster Translation ArcStar Feature Track", 66000*4)
    #drawFeatureTrack3D_New(efast_events_factor, "Poster Translation eFast Feature Track", 66000*4)

    #sae = np.zeros((HEIGHT, WIDTH, 2), dtype=np.int64)
    #prev_time = 0
    e_count, a_count = 0, 0
    for i, (t, x, y, p) in tqdm(enumerate(all_events)):
        #sae = update_time_surface_new(sae, t, x, y, p, mode="absolute")
        #sae = update_time_surface_new(sae, t, x, y, p, time_threshold=ACCUMULATE_TIME, prev_time=prev_time, mode="delta")
        #prev_time = t
        
        #if ((t, x, y, p) in efast_events) and ((t, x, y, p) not in efast_events_66ms):
        #if ((t, x, y, p) in efast_events) and ((t, x, y, p) not in efast_events_33ms):
        if ((t, x, y, p) in efast_events_factor) and ((t, x, y, p) in efast_events_abs):
        #if ((t, x, y, p) in efast_events_factor) and ((t, x, y, p) in efast_events):
        #if ((t, x, y, p) in efast_events_factor):
        #if ((t, x, y, p) in efast_events):
            #img = crop(sae[:, :, p], x, y, 9)

            #mask = np.where(img > 0)
            #img_min = img[mask].min()
            #img_max = img[mask].max()

            #fig = plt.figure()
            #ax = fig.add_subplot(1,1,1)
            #ax.imshow(img, cmap='coolwarm')
            #shw = ax.imshow(img, cmap='coolwarm', vmin=img_min, vmax=img_max)
            #bar = plt.colorbar(shw)
            #plt.savefig("./Output/eFast/"+str(t)+"-"+str(x)+"-"+str(y)+".jpg", dpi=100)
            #plt.close(fig)
            
            e_count += 1
        
        #if ((t, x, y, p) in arcstar_events) and ((t, x, y, p) not in arcstar_events_66ms):
        #if ((t, x, y, p) in arcstar_events) and ((t, x, y, p) not in arcstar_events_33ms):
        if ((t, x, y, p) in arcstar_events_factor) and ((t, x, y, p) in arcstar_events_abs):
        #if ((t, x, y, p) in arcstar_events_factor) and ((t, x, y, p) in arcstar_events):
        #if ((t, x, y, p) in arcstar_events_factor):
        #if ((t, x, y, p) in arcstar_events):
            #img = crop(sae[:, :, p], x, y, 9)

            #mask = np.where(img > 0)
            #img_min = img[mask].min()
            #img_max = img[mask].max()

            #fig = plt.figure()
            #ax = fig.add_subplot(1,1,1)
            #ax.imshow(img, cmap='coolwarm')
            #shw = ax.imshow(img, cmap='coolwarm', vmin=img_min, vmax=img_max)
            #bar = plt.colorbar(shw)
            #plt.savefig("./Output/ArcStar/"+str(t)+"-"+str(x)+"-"+str(y)+".jpg", dpi=100)
            #plt.close(fig)
            
            a_count += 1
        
        """
        if (t, x, y, p) in efast_events:
            #efast_count += 1
            img = crop(sae[:, :, p], x, y, 9)

            mask = np.ma.masked_equal(img, 0, copy=False)
            img_min = mask.min()
            img -= img_min
            mask = np.where(img < 0)
            img[mask] = 0

            fig = plt.figure()
            ax = fig.add_subplot(1,1,1)
            ax.imshow(img, cmap='coolwarm')
            plt.savefig("./Output/eFast/"+str(t)+"-"+str(x)+"-"+str(y)+".jpg", dpi=100)
            plt.close(fig)
        
        if (t, x, y, p) in arcstar_events:
            #arcstar_count += 1
            img = crop(sae[:, :, p], x, y, 9)

            mask = np.ma.masked_equal(img, 0, copy=False)
            img_min = mask.min()
            img -= img_min
            mask = np.where(img < 0)
            img[mask] = 0

            fig = plt.figure()
            ax = fig.add_subplot(1,1,1)
            ax.imshow(img, cmap='coolwarm')
            plt.savefig("./Output/ArcStar/"+str(t)+"-"+str(x)+"-"+str(y)+".jpg", dpi=100)
            plt.close(fig)
        """
    print(e_count, a_count)
    print("Finished!!!")
