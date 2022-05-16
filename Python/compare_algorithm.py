from tqdm import tqdm
import numpy as np
import os

from src.util import *
from src.plot_tools import *
from src.process_events import *

ACCUMULATE_TIME = 33000
RESULT_FOLDERS = ["./New_33000us_100000/", "./New_33000us_400000/", "./New_66000us_100000/", "./New_66000us_400000/", "./New_Absolute_Result_100000/", "./New_Absolute_Result_400000/"]
ORIGINAL_TIMESTAMP = "./Rotation_Absolute/"
ORIGINAL_66000 = "./Rotation_66000/"
ORIGINAL_33000 = "./Rotation_33000/"
ALL_EVENT_FILE = "All_events.txt"
ALL_ARCSTAR_FILE = "arcStar_Corners.txt"
ALL_EFAST_FILE = "eFast_Corners.txt"

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
    plt.show()
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

if __name__ == "__main__":
    # Compare result against original algo.
    # First we load the corner detected by the Original with Timestamp
    original_timestamp_efast = extract_events(ORIGINAL_TIMESTAMP+ALL_EFAST_FILE)
    original_timestamp_arcstar = extract_events(ORIGINAL_TIMESTAMP+ALL_ARCSTAR_FILE)
    # Next we load the 400000 version of our new algo
    new_timestamp_efast = extract_events(RESULT_FOLDERS[5]+ALL_EFAST_FILE)
    new_timestamp_arcstar = extract_events(RESULT_FOLDERS[5]+ALL_ARCSTAR_FILE)

    original_66000_efast = extract_events(ORIGINAL_66000+ALL_EFAST_FILE)
    original_66000_arcstar = extract_events(ORIGINAL_66000+ALL_ARCSTAR_FILE)
    # Next we load the 400000 version of our new algo
    new_66000_efast = extract_events(RESULT_FOLDERS[3]+ALL_EFAST_FILE)
    new_66000_arcstar = extract_events(RESULT_FOLDERS[3]+ALL_ARCSTAR_FILE)

    original_33000_efast = extract_events(ORIGINAL_33000+ALL_EFAST_FILE)
    original_33000_arcstar = extract_events(ORIGINAL_33000+ALL_ARCSTAR_FILE)
    # Next we load the 400000 version of our new algo
    new_33000_efast = extract_events(RESULT_FOLDERS[1]+ALL_EFAST_FILE)
    new_33000_arcstar = extract_events(RESULT_FOLDERS[1]+ALL_ARCSTAR_FILE)

    if not os.path.isdir("./Output/eFast/"):
        os.makedirs("./Output/eFast/")
    if not os.path.isdir("./Output/ArcStar/"):
        os.makedirs("./Output/ArcStar/")
    if not os.path.isdir("../Output/"):
        os.makedirs("../Output/")

    """
        Calculate the Performance of using Timestamp SAE
    """

    tp, fp = 0, 0
    for i, (t, x, y, p) in tqdm(enumerate(new_timestamp_efast)):
        # Calculate number of overlapping corners. These are the "True Positives"
        if (t, x, y, p) in original_timestamp_efast:
            tp += 1
        # Otherwise we will classify the rest as "False Positive"
        else:
            fp += 1

    print("Result of the Timestamp eFast Comparison:")
    print("\tNumber of eFast Corner of the Original Algo: " + str(len(original_timestamp_efast)))
    print("\tNumber of eFast Corner of the New Algo: " + str(len(new_timestamp_efast)))
    print("\tNumber of overlapping Corners are: " + str(tp))
    print("\tNumber of unique corners detected by the new algo: " + str(fp))
    print("\tNumber of unique corners detected by the old algo: " + str(len(original_timestamp_efast) - tp))
    print("-------------------------------------------------------")

    tp, fp = 0, 0
    for i, (t, x, y, p) in tqdm(enumerate(new_timestamp_arcstar)):
        # Calculate number of overlapping corners. These are the "True Positives"
        if (t, x, y, p) in original_timestamp_arcstar:
            tp += 1
        # Otherwise we will classify the rest as "False Positive"
        else:
            fp += 1
    
    print("Result of the Timestamp arc* Comparison:")
    print("\tNumber of arc* Corner of the Original Algo: " + str(len(original_timestamp_arcstar)))
    print("\tNumber of arc* Corner of the New Algo: " + str(len(new_timestamp_arcstar)))
    print("\tNumber of overlapping Corners are: " + str(tp))
    print("\tNumber of unique corners detected by the new algo: " + str(fp))
    print("\tNumber of unique corners detected by the old algo: " + str(len(original_timestamp_arcstar) - tp))
    print("-------------------------------------------------------")

    """
        Calculate the Performance of using 66000us SAE
    """

    tp, fp = 0, 0
    for i, (t, x, y, p) in tqdm(enumerate(new_66000_efast)):
        # Calculate number of overlapping corners. These are the "True Positives"
        if (t, x, y, p) in original_66000_efast:
            tp += 1
        # Otherwise we will classify the rest as "False Positive"
        else:
            fp += 1

    print("Result of the 66000 eFast Comparison:")
    print("\tNumber of eFast Corner of the Original Algo: " + str(len(original_66000_efast)))
    print("\tNumber of eFast Corner of the New Algo: " + str(len(new_66000_efast)))
    print("\tNumber of overlapping Corners are: " + str(tp))
    print("\tNumber of unique corners detected by the new algo: " + str(fp))
    print("\tNumber of unique corners detected by the old algo: " + str(len(original_66000_efast) - tp))
    print("-------------------------------------------------------")

    tp, fp = 0, 0
    for i, (t, x, y, p) in tqdm(enumerate(new_66000_arcstar)):
        # Calculate number of overlapping corners. These are the "True Positives"
        if (t, x, y, p) in original_66000_arcstar:
            tp += 1
        # Otherwise we will classify the rest as "False Positive"
        else:
            fp += 1
    
    print("Result of the 66000 arc* Comparison:")
    print("\tNumber of arc* Corner of the Original Algo: " + str(len(original_66000_arcstar)))
    print("\tNumber of arc* Corner of the New Algo: " + str(len(new_66000_arcstar)))
    print("\tNumber of overlapping Corners are: " + str(tp))
    print("\tNumber of unique corners detected by the new algo: " + str(fp))
    print("\tNumber of unique corners detected by the old algo: " + str(len(original_66000_arcstar) - tp))
    print("-------------------------------------------------------")

    """
        Calculate the Performance of using 33000us SAE
    """

    tp, fp = 0, 0
    for i, (t, x, y, p) in tqdm(enumerate(new_33000_efast)):
        # Calculate number of overlapping corners. These are the "True Positives"
        if (t, x, y, p) in original_33000_efast:
            tp += 1
        # Otherwise we will classify the rest as "False Positive"
        else:
            fp += 1

    print("Result of the 33000 eFast Comparison:")
    print("\tNumber of eFast Corner of the Original Algo: " + str(len(original_33000_efast)))
    print("\tNumber of eFast Corner of the New Algo: " + str(len(new_33000_efast)))
    print("\tNumber of overlapping Corners are: " + str(tp))
    print("\tNumber of unique corners detected by the new algo: " + str(fp))
    print("\tNumber of unique corners detected by the old algo: " + str(len(original_33000_efast) - tp))
    print("-------------------------------------------------------")

    tp, fp = 0, 0
    for i, (t, x, y, p) in tqdm(enumerate(new_33000_arcstar)):
        # Calculate number of overlapping corners. These are the "True Positives"
        if (t, x, y, p) in original_33000_arcstar:
            tp += 1
        # Otherwise we will classify the rest as "False Positive"
        else:
            fp += 1
    
    print("Result of the 33000 arc* Comparison:")
    print("\tNumber of arc* Corner of the Original Algo: " + str(len(original_33000_arcstar)))
    print("\tNumber of arc* Corner of the New Algo: " + str(len(new_33000_arcstar)))
    print("\tNumber of overlapping Corners are: " + str(tp))
    print("\tNumber of unique corners detected by the new algo: " + str(fp))
    print("\tNumber of unique corners detected by the old algo: " + str(len(original_33000_arcstar) - tp))
    print("-------------------------------------------------------")

    drawFeatureTrack3D_New(new_timestamp_efast, "Shapes Rotation eFast Feature Track", 1)
    drawFeatureTrack3D_New(new_timestamp_arcstar, "Shapes Rotation Arcstar Feature Track", 1)
    drawFeatureTrack3D_New(new_66000_efast, "Shapes Rotation eFast Feature Track", 66000)
    drawFeatureTrack3D_New(new_66000_arcstar, "Shapes Rotation ArcStar Feature Track", 66000)
    drawFeatureTrack3D_New(new_33000_efast, "Shapes Rotation eFast Feature Track", 33000)
    drawFeatureTrack3D_New(new_33000_arcstar, "Shapes Rotation ArcStar Feature Track", 33000)

    """
    sae = np.zeros((HEIGHT, WIDTH, 2), dtype=np.int64)
    prev_time = 0
    for i, (t, x, y, p) in tqdm(enumerate(all_events)):
        #sae = update_time_surface(sae, t, x, y, p, mode="absolute")
        sae = update_time_surface(sae, t, x, y, p, time_threshold=ACCUMULATE_TIME, prev_time=prev_time, mode="delta")
        prev_time = t

        if i >= 20000 and i <= 60000:
            #if ((t, x, y, p) in efast_events) and ((t, x, y, p) not in efast_events_66ms):
            if ((t, x, y, p) in efast_events) and ((t, x, y, p) not in efast_events_33ms):
                img = crop(sae[:, :, p], x, y, 9)
                fig = plt.figure()
                ax = fig.add_subplot(1,1,1)
                ax.imshow(img, cmap='coolwarm')
                plt.savefig("./Output/eFast/"+str(t)+"-"+str(x)+"-"+str(y)+".jpg", dpi=100)
                plt.close(fig)
            
            #if ((t, x, y, p) in arcstar_events) and ((t, x, y, p) not in arcstar_events_66ms):
            if ((t, x, y, p) in arcstar_events) and ((t, x, y, p) not in arcstar_events_33ms):
                img = crop(sae[:, :, p], x, y, 9)
                fig = plt.figure()
                ax = fig.add_subplot(1,1,1)
                ax.imshow(img, cmap='coolwarm')
                plt.savefig("./Output/ArcStar/"+str(t)+"-"+str(x)+"-"+str(y)+".jpg", dpi=100)
                plt.close(fig)
    """

    """
        if (t, x, y, p) in efast_events:
            efast_count += 1
            img = crop(sae[:, :, p], x, y, 9)
            fig = plt.figure()
            ax = fig.add_subplot(1,1,1)
            ax.imshow(img, cmap='coolwarm')
            plt.savefig("./Output/eFast/"+str(t)+"-"+str(x)+"-"+str(y)+".jpg", dpi=100)
            plt.close(fig)
        
        if (t, x, y, p) in arcstar_events:
            arcstar_count += 1
            img = crop(sae[:, :, p], x, y, 9)
            fig = plt.figure()
            ax = fig.add_subplot(1,1,1)
            ax.imshow(img, cmap='coolwarm')
            plt.savefig("./Output/ArcStar/"+str(t)+"-"+str(x)+"-"+str(y)+".jpg", dpi=100)
            plt.close(fig)
    """

    print("Finished!!!")
