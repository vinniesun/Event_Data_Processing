from math import ceil, sqrt
import os
from src.efast import *
from src.arcstar import *
from src.util import *
from src.harris_detector import *
from src.plot_tools import *
from src.process_events import *

aedat4_file = '../EBBINNOT_AEDAT4/Recording/20180711_Site1_3pm_12mm_01.aedat4'

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
    current_events.events, current_events.num_events = background_activity_filter(current_events, time_window=33000)
    print("After NN filtering, number of events remaining is: ", current_events.num_events)
    print("---------------------------------------------------------------")

    # Generate Time Surface
    sae = generate_time_surface(current_events, 33000, event_start=102342-12000, event_end=102343-6000)

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

        sae = update_time_surface(sae, 33000, t, x, y, p, prev_time)
        image = sae[:, :, int(p)]

        # eFast Corner
        isEFast = isCornerEFast(image, x, y, int(p))
        # Arc* Corner
        isArcStar = isCornerArcStar(image, prev_state, prev_state_inv, x, y, int(p))

        if isEFast and isArcStar:
            pastEventQueue.append((x, y, 255))
            allEFastQueue.append((image.copy(), x, y, i, p))
            allArcStarQueue.append((image.copy(), x, y, i, p))
            bothQueue.append((image.copy(), x, y, i, p))
        elif isEFast:
            pastEventQueue.append((x, y, 255))
            allEFastQueue.append((image.copy(), x, y, i, p))
            eFastQueue.append((image.copy(), x, y, i, p))
        elif isArcStar:
            pastEventQueue.append((x, y, 255))
            ArcStarQueue.append((image.copy(), x, y, i, p))
            allArcStarQueue.append((image.copy(), x, y, i, p))
        else:
            pastEventQueue.append((x, y, 0))

    image = sae[:, :, int(p)]

    print(on_count, off_count)
    # Draw the Current SAE
    drawHeatMapWhole(image, 102343, True, name="SAE" + str(33000) + "us")

    # Plot Feature Track of last 3000 Events
    drawFeatureTrack3D(pastEventQueue, name="All Event Feature Track", time_step=33000)
    drawFeatureTrack2D(pastEventQueue, name="All Event Feature Track", time_step=33000)

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
    plt.savefig("../Output/" + str(33000) + "us 2D eFast Only, Corners detected - " + str(len(eFastQueue)) + ".jpg", dpi=300)
    plt.show()
    plt.close(fig)
    
    drawEventLocation(SAELocation, i, True, name="Only eFast SAE Location" + str(33000) + "us", cornerLocation=eFastQueue[-20:])

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
    plt.savefig("../Output/" + str(33000) + "us 2D ArcStar Only, Corners detected - " + str(len(ArcStarQueue)) + ".jpg", dpi=300)
    plt.show()
    plt.close(fig)
    
    drawEventLocation(SAELocation, i, True, name="Only ArcStar SAE Location" + str(33000) + "us", cornerLocation=ArcStarQueue[-20:])

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
    plt.savefig("../Output/" + str(33000) + "us 2D Both, Corners detected - " + str(len(bothQueue)) + ".jpg", dpi=300)
    plt.show()
    plt.close(fig)
    
    drawEventLocation(SAELocation, i, True, name="Both SAE Location" + str(33000) + "us", cornerLocation=bothQueue[-20:])

    # Plot all corners detected by eFast
    SAELocation = np.ones((HEIGHT, WIDTH), dtype=np.uint8)
    SAELocation *= 128
    for j, (img, locX, locY, recEventNo, p) in enumerate(allEFastQueue):
        if p:
            SAELocation[locY][locX] = 255 # for On Events
        else:
            SAELocation[locY][locX] = 0 # for Off Events
    drawEventLocation(SAELocation, i, True, name="All eFast SAE Location" + str(33000) + "us", cornerLocation=allEFastQueue)

    # Plot all corners detected by ArcStar
    SAELocation = np.ones((HEIGHT, WIDTH), dtype=np.uint8)
    SAELocation *= 128
    for j, (img, locX, locY, recEventNo, p) in enumerate(allArcStarQueue):
        if p:
            SAELocation[locY][locX] = 255 # for On Events
        else:
            SAELocation[locY][locX] = 0 # for Off Events
    drawEventLocation(SAELocation, i, True, name="All ArcStar SAE Location" + str(33000) + "us", cornerLocation=allArcStarQueue)

    # Plot all corners detected by ArcStar
    SAELocation = np.ones((HEIGHT, WIDTH), dtype=np.uint8)
    SAELocation *= 128
    for j, (img, locX, locY, recEventNo, p) in enumerate(bothQueue):
        if p:
            SAELocation[locY][locX] = 255 # for On Events
        else:
            SAELocation[locY][locX] = 0 # for Off Events
    drawEventLocation(SAELocation, i, True, name="All of Both SAE Location" + str(33000) + "us", cornerLocation=bothQueue)

if __name__ == "__main__":
    if not os.path.isdir("../Output"):
        os.mkdir("../Output")

    main()

    #read_aedat4(aedat4_file, time_step=33000, processAll=False, eventSpecified=True, eventNo=102342) # 33000ms interval, @event=54000, we get arc* response, no for eFast
    # 5000ms interval, @eventNo 44000, eFast response, no arc* response
    # 67239
