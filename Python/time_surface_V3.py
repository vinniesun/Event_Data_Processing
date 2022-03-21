from math import ceil, sqrt
import os
from src.efast import *
from src.arcstar import *
from src.util import *
from src.harris_detector import *
from src.plot_tools import *
from src.process_events import *

aedat4_file = '../EBBINNOT_AEDAT4/Recording/20180711_Site1_3pm_12mm_01.aedat4'

def drawFeatureTrack2D_New(pastEventQueue, name, time_step):
    fig = plt.figure(figsize=(16,10))
    c = ["red","lightcoral","black", "palegreen","green"]
    v = [0,.4,.5,0.6,1.]
    l = list(zip(v,c))
    myCMap=LinearSegmentedColormap.from_list('rg',l, N=256)

    xData = []
    yData = []
    cData = []
    for i, (_, locX, locY, t, corner) in enumerate(pastEventQueue):
        xData.append(t)
        yData.append(locX)
        cData.append(corner)

    plt.scatter(xData, yData, s=1, c=cData, cmap=myCMap)
    plt.xlabel("time")
    plt.ylabel("width")

    plt.savefig("../Output/2D Event Feature Track of " + name + str(time_step) + ".jpg", dpi=300)
    plt.show()
    plt.close()

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
    for i, (_, locX, locY, t, corner) in enumerate(pastEventQueue):
        xData.append(locX)
        yData.append(t)
        zData.append(locY)
        cData.append(corner)

    ax.scatter3D(xData, yData, zData, s=1, c=cData, cmap=myCMap)
    ax.set_xlabel("width")
    ax.set_ylabel("time")
    ax.set_zlabel("height")

    plt.savefig("../Output/3D Event Feature Track of " + name + str(time_step) + ".jpg", dpi=300)
    plt.show()
    plt.close()

def main() -> None:
    # Load file and extract all events
    current_events = aedat_to_events(aedat4_file)
    print("Starting number of event is: ", current_events.num_events)
    print("---------------------------------------------------------------")

    # Apply Refractory Filtering:
    current_events.events, current_events.num_events = refractory_filtering(current_events, refractory_period=100)
    print("After refractory filtering, number of events remaining is: ", current_events.num_events)
    print("---------------------------------------------------------------")

    
    # Apply Background Activity Filtering
    current_events.events, current_events.num_events = background_activity_filter(current_events, time_window=5000) # Try 1ms~5ms
    print("After NN filtering, number of events remaining is: ", current_events.num_events)
    print("---------------------------------------------------------------")

    # Generate Time Surface
    sae = generate_time_surface(current_events, 33000, event_start=102342-12000, event_end=102343-6000, mode="delta")
    #sae = np.zeros((current_events.height, current_events.width, 2), dtype=np.int64)

    #p = int(current_events.events[102343]["p"])
    #image = sae[:, :, int(p)]

    # Draw the Current SAE
    #drawHeatMapWhole(image, 102343, True, name="SAE" + str(33000) + "us")

    eFastQueue, allEFastQueue = [], []
    ArcStarQueue, allArcStarQueue = [], []
    bothQueue = []
    pastEventQueue = []

    prev_state, prev_state_inv = 0, 0
    prev_time = 0

    on_count, off_count = 0, 0
    # Generate corners
    for i in tqdm(range(102342-6000, 102343)):
        prev_time = current_events.events[i-1]["t"]
        t, x, y, p = current_events.events[i]["t"], current_events.events[i]["x"], current_events.events[i]["y"], current_events.events[i]["p"]

        if p:
            on_count += 1
        else:
            off_count += 1

        prev_state = sae[y][x][int(p)] # This is for ArcStar
        prev_state_inv = sae[y][x][int(not p)] # This is for ArcStar

        sae = update_time_surface(sae, t, x, y, p, 33000, prev_time=prev_time, mode="delta")
        image = sae[:, :, int(p)]

        #prev_time = t

        # eFast Corner
        isEFast = isCornerEFast(image, x, y, int(p))
        # Arc* Corner
        isArcStar = isCornerArcStar(image, prev_state, prev_state_inv, x, y, int(p))

        if isEFast and isArcStar:
            pastEventQueue.append((1, x, y, t, 255))
            allEFastQueue.append((image.copy(), x, y, t, p))
            allArcStarQueue.append((image.copy(), x, y, t, p))
            bothQueue.append((image.copy(), x, y, t, p))
        elif isEFast:
            pastEventQueue.append((1, x, y, t, 255))
            allEFastQueue.append((image.copy(), x, y, t, p))
            eFastQueue.append((image.copy(), x, y, t, p))
        elif isArcStar:
            pastEventQueue.append((1, x, y, t, 255))
            ArcStarQueue.append((image.copy(), x, y, t, p))
            allArcStarQueue.append((image.copy(), x, y, t, p))
        else:
            pastEventQueue.append((1, x, y, t, 0))

    image = sae[:, :, int(p)]

    print(on_count, off_count)
    # Draw the Current SAE
    drawHeatMapWhole(image, 102343, True, name="SAE" + str(5000) + "us")

    # Plot Feature Track of last 3000 Events
    drawFeatureTrack3D_New(pastEventQueue, name="All Event Feature Track", time_step=5000)
    drawFeatureTrack2D_New(pastEventQueue, name="All Event Feature Track", time_step=5000)

    # Plot eFast corner Tracks of last 3000 Events
    drawFeatureTrack3D_New(allEFastQueue, name="All eFast Corner Feature Track", time_step=5000)
    drawFeatureTrack2D_New(allEFastQueue, name="All eFast Corner Feature Track", time_step=5000)

    # Plot Arc* corner Tracks of last 3000 Events
    drawFeatureTrack3D_New(allArcStarQueue, name="All ArcStar Corner Feature Track", time_step=5000)
    drawFeatureTrack2D_New(allArcStarQueue, name="All ArcStar Corner Feature Track", time_step=5000)

    # Plot Both corner Tracks of last 3000 Events
    drawFeatureTrack3D_New(bothQueue, name="All Both Corner Feature Track", time_step=5000)
    drawFeatureTrack2D_New(bothQueue, name="All Both Corner Feature Track", time_step=5000)
    print("number of eFast: " + str(len(eFastQueue)))
    print("number of ArcStar: " + str(len(ArcStarQueue)))
    print("number of Both: " + str(len(bothQueue)))

    # Plot All eFast Response
    noRow, noCol = 5, 4
    SAELocation = np.ones((HEIGHT, WIDTH), dtype=np.uint8)
    SAELocation *= 128
    fig = plt.figure(figsize=(30, 30))
    plt.tight_layout()
    for j, (img, locX, locY, recEventNo, p) in enumerate(eFastQueue[-20:]):
        if p:
            SAELocation[locY][locX] = 255 # for On Events
        else:
            SAELocation[locY][locX] = 0 # for Off Events
        ax = fig.add_subplot(noRow, noCol, j+1) 
        temp = crop(img, locX, locY, 9)
        ax = drawHeatMapSub(temp, recEventNo, subplot=ax, title=str(recEventNo) + "\n" + str(locY) + ", " + str(locX))
    plt.savefig("../Output/" + str(5000) + "us 2D eFast Only, Corners detected - " + str(len(eFastQueue)) + ".jpg", dpi=300)
    plt.show()
    plt.close(fig)
    
    drawEventLocation(SAELocation, i, True, name="Only eFast SAE Location" + str(5000) + "us", cornerLocation=eFastQueue[-20:])

    # Plot All ArcStar Response
    noRow, noCol = 5, 4
    SAELocation = np.ones((HEIGHT, WIDTH), dtype=np.uint8)
    SAELocation *= 128
    fig = plt.figure(figsize=(30, 30))
    plt.tight_layout()
    for j, (img, locX, locY, recEventNo, p) in enumerate(ArcStarQueue[-20:]):
        if p:
            SAELocation[locY][locX] = 255 # for On Events
        else:
            SAELocation[locY][locX] = 0 # for Off Events
        ax = fig.add_subplot(noRow, noCol, j+1)
        temp = crop(img, locX, locY, 9)
        ax = drawHeatMapSub(temp, recEventNo, subplot=ax, title=str(recEventNo) + "\n" + str(locY) + ", " + str(locX))
    plt.savefig("../Output/" + str(5000) + "us 2D ArcStar Only, Corners detected - " + str(len(ArcStarQueue)) + ".jpg", dpi=300)
    plt.show()
    plt.close(fig)
    
    drawEventLocation(SAELocation, i, True, name="Only ArcStar SAE Location" + str(5000) + "us", cornerLocation=ArcStarQueue[-20:])

    # Plot Both
    noRow, noCol = 5, 4
    SAELocation = np.ones((HEIGHT, WIDTH), dtype=np.uint8)
    SAELocation *= 128
    fig = plt.figure(figsize=(30, 30))
    plt.tight_layout()
    for j, (img, locX, locY, recEventNo, p) in enumerate(bothQueue[-20:]):
        if p:
            SAELocation[locY][locX] = 255 # for On Events
        else:
            SAELocation[locY][locX] = 0 # for Off Events
        ax = fig.add_subplot(noRow, noCol, j+1)
        temp = crop(img, locX, locY, 9)
        ax = drawHeatMapSub(temp, recEventNo, subplot=ax, title=str(recEventNo) + "\n" + str(locY) + ", " + str(locX))
    plt.savefig("../Output/" + str(5000) + "us 2D Both, Corners detected - " + str(len(bothQueue)) + ".jpg", dpi=300)
    plt.show()
    plt.close(fig)
    
    drawEventLocation(SAELocation, i, True, name="Both SAE Location" + str(5000) + "us", cornerLocation=bothQueue[-20:])

    # Plot all corners detected by eFast
    SAELocation = np.ones((HEIGHT, WIDTH), dtype=np.uint8)
    SAELocation *= 128
    for j, (img, locX, locY, recEventNo, p) in enumerate(allEFastQueue):
        if p:
            SAELocation[locY][locX] = 255 # for On Events
        else:
            SAELocation[locY][locX] = 0 # for Off Events
    drawEventLocation(SAELocation, i, True, name="All eFast SAE Location" + str(5000) + "us", cornerLocation=allEFastQueue)
    print("Total number of eFast corners are: " + str(len(allEFastQueue)))

    # Plot all corners detected by ArcStar
    SAELocation = np.ones((HEIGHT, WIDTH), dtype=np.uint8)
    SAELocation *= 128
    for j, (img, locX, locY, recEventNo, p) in enumerate(allArcStarQueue):
        if p:
            SAELocation[locY][locX] = 255 # for On Events
        else:
            SAELocation[locY][locX] = 0 # for Off Events
    drawEventLocation(SAELocation, i, True, name="All ArcStar SAE Location" + str(5000) + "us", cornerLocation=allArcStarQueue)
    print("Total number of ArcStar corners are: " + str(len(allArcStarQueue)))

    # Plot all corners detected by ArcStar
    SAELocation = np.ones((HEIGHT, WIDTH), dtype=np.uint8)
    SAELocation *= 128
    for j, (img, locX, locY, recEventNo, p) in enumerate(bothQueue):
        if p:
            SAELocation[locY][locX] = 255 # for On Events
        else:
            SAELocation[locY][locX] = 0 # for Off Events
    drawEventLocation(SAELocation, i, True, name="All of Both SAE Location" + str(5000) + "us", cornerLocation=bothQueue)
    print("Total number of Both corners are: " + str(len(bothQueue)))

if __name__ == "__main__":
    if not os.path.isdir("../Output"):
        os.mkdir("../Output")

    main()

    #read_aedat4(aedat4_file, time_step=33000, processAll=False, eventSpecified=True, eventNo=102342) # 33000ms interval, @event=54000, we get arc* response, no for eFast
    # 5000ms interval, @eventNo 44000, eFast response, no arc* response
    # 67239
