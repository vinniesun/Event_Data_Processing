from math import ceil, sqrt
import aedat
import os
from tqdm import tqdm
from src.efast import *
from src.arcstar import *
from src.util import *
from src.harris_detector import *
from src.plot_tools import *
from src.process_events import background_activity_filter

aedat4_file = '../EBBINNOT_AEDAT4/Recording/20180711_Site1_3pm_12mm_01.aedat4'
dt = 1

def read_aedat4(filename:str, time_step:int=66000, processAll:bool=True, factor:int=1, eventSpecified:bool=False, eventNo:int=None) -> None:
    data = aedat.Decoder(filename)
    for packet in data:
        refractory_period = 2000
        total_events = len(packet['events'])
        noFrames = ceil(ceil(packet['events']['t'][-1]/time_step))
        sae = np.zeros((HEIGHT, WIDTH, 2), dtype=np.int64)
        last_time = np.zeros((HEIGHT, WIDTH), dtype=np.int64) - refractory_period
        #last_time = np.zeros((HEIGHT, WIDTH, 2), dtype=np.int64)
        ebbi_image = np.zeros((noFrames, HEIGHT, WIDTH, 2), dtype=np.uint8)

        last_event = 0
        
        pastOnEvents = []
        pastOffEvents = []
        pastEventQueue = []

        threshold_val = time_step

        frame = 0
        accumulated_time = 0

        eFastQueue = []
        ArcStarQueue = []
        bothQueue = []
        allEFastQueue = []
        allArcStarQueue = []

        eventCount = 0
        x_prev, y_prev, p_prev = 0, 0, 0
        valid_index = 0

        for i in tqdm(range(0, total_events, dt), desc='processing events'):
            y = packet['events']['y'][i]
            x = packet['events']['x'][i]
            pol = packet['events']['on'][i] # packet['events']['on'] = 1 means it's ON, packet['event']['on'] = 0 means it's OFF

            # Apply refraction
            if packet['events']['t'][i] - last_time[y][x] < refractory_period:
                last_time[y][x] = packet['events']['t'][i]
                continue

            last_time[y][x] = packet['events']['t'][i]
            prev_state = sae[y][x][int(pol)] # This is for ArcStar
            prev_state_inv = sae[y][x][int(not pol)] # This is for ArcStar

            deltaT = int(packet['events']['t'][i] - last_event)

            sae[:, :, int(pol)] -= deltaT
            sae[y][x][int(pol)] = threshold_val
            temp = sae[:, :, int(pol)]
            temp[temp < 0] = 0
            sae[:, :, int(pol)] = temp

            ebbi_image[frame][y][x][int(pol)] = 255
            
            if pol:
                if i >= time_step:
                    eventRemoved = pastOnEvents[0]
                    pastOnEvents.pop(0)
                    ebbi_image[frame][eventRemoved[0]][eventRemoved[1]][int(pol)] = 0
                    pastOnEvents.append((y, x))
                else:
                    pastOnEvents.append((y, x))
            else:
                if i >= time_step:
                    eventRemoved = pastOffEvents[0]
                    pastOffEvents.pop(0)
                    ebbi_image[frame][eventRemoved[0]][eventRemoved[1]][int(pol)] = 0
                    pastOffEvents.append((y, x))
                else:
                    pastOffEvents.append((y, x))
            

            image = sae[:, :, int(pol)]
            if eventNo - 3000 < i <= eventNo:
                isEFast = isCornerEFast(image, x, y, int(pol))
                isArcStar = isCornerArcStar(image, prev_state, prev_state_inv, x, y, int(pol))
                if isEFast and isArcStar:
                    pastEventQueue.append((x, y, 255))
                    allEFastQueue.append((image.copy(), x, y, i, pol))
                    allArcStarQueue.append((image.copy(), x, y, i, pol))
                    bothQueue.append((image.copy(), x, y, i, pol))
                elif isEFast:
                    pastEventQueue.append((x, y, 255))
                    allEFastQueue.append((image.copy(), x, y, i, pol))
                    eFastQueue.append((image.copy(), x, y, i, pol))
                elif isArcStar:
                    pastEventQueue.append((x, y, 255))
                    ArcStarQueue.append((image.copy(), x, y, i, pol))
                    allArcStarQueue.append((image.copy(), x, y, i, pol))
                else:
                    pastEventQueue.append((x, y, 0))
            
            if i == eventNo:
            #if isCornerEFast(image, x, y, int(pol)) and not isCornerArcStar(image, prev_state, prev_state_inv, x, y, int(pol)) and i >= 5000:
                #filtered = medianFilter(image, 3)
                #image = filtered
                print("length of past events", len(pastEventQueue))
                
                # Draw the Current EBBI Image
                drawHeatMapWhole(ebbi_image[frame][:, :, int(pol)], i, eventSpecified, name="EBBI Image" + str(time_step) + "us")

                # Draw the Current SAE
                drawHeatMapWhole(image, i, eventSpecified, name="SAE" + str(time_step) + "us")

                noCorner, output, cornerLocation = openCVHarrisCornerDet(ebbi_image[frame][:, :, int(pol)], x, y, int(pol))
                #noCorner, output, cornerLocation = customHarrisCornerDet(ebbi_image[frame][:, :, int(pol)], x, y, int(pol))
                drawHeatMapWhole(np.zeros((output.shape[0], output.shape[1])), i, eventSpecified, name="All Corners " + str(time_step) + "us", symbol="x", cornerLocation=cornerLocation)
                noRow, noCol = int((sqrt(noCorner))), ceil(noCorner / int(sqrt(noCorner)))

                # Plot Harris Response
                fig, ax = plt.subplots(noRow, noCol,figsize=(16, 16))
                plt.tight_layout()
                for j, (row, col) in enumerate(cornerLocation):
                    temp = crop(image, col, row, 9)
                    ax[j // noCol][j % noCol] = drawHeatMapSub(temp, i, subplot=ax[j // noCol][j % noCol], title="Corner " + str(row) + ", " + str(col))
                plt.savefig("../Output/" + str(time_step) + "us 2D Harris, Corners detected - " + str(noCorner) + ", of Event " + str(i) + ".jpg", dpi=300)
                plt.show()
                plt.close(fig)

                # Plot Feature Track of last 3000 Events
                drawFeatureTrack3D(pastEventQueue, name="All Event Feature Track", time_step=time_step)
                drawFeatureTrack2D(pastEventQueue, name="All Event Feature Track", time_step=time_step)

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
                plt.savefig("../Output/" + str(time_step) + "us 2D eFast Only, Corners detected - " + str(len(eFastQueue)) + ".jpg", dpi=300)
                plt.show()
                plt.close(fig)
                
                drawEventLocation(SAELocation, i, eventSpecified, name="Only eFast SAE Location" + str(time_step) + "us", cornerLocation=eFastQueue[-20:])

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
                plt.savefig("../Output/" + str(time_step) + "us 2D ArcStar Only, Corners detected - " + str(len(ArcStarQueue)) + ".jpg", dpi=300)
                plt.show()
                plt.close(fig)
                
                drawEventLocation(SAELocation, i, eventSpecified, name="Only ArcStar SAE Location" + str(time_step) + "us", cornerLocation=ArcStarQueue[-20:])

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
                plt.savefig("../Output/" + str(time_step) + "us 2D Both, Corners detected - " + str(len(bothQueue)) + ".jpg", dpi=300)
                plt.show()
                plt.close(fig)
                
                drawEventLocation(SAELocation, i, eventSpecified, name="Both SAE Location" + str(time_step) + "us", cornerLocation=bothQueue[-20:])

                SAELocation = np.ones((HEIGHT, WIDTH), dtype=np.uint8)
                SAELocation *= 128
                for j, (img, locX, locY, recEventNo, p) in enumerate(allEFastQueue):
                    if p:
                        SAELocation[locY][locX] = 255 # for On Events
                    else:
                        SAELocation[locY][locX] = 0 # for Off Events
                drawEventLocation(SAELocation, i, eventSpecified, name="All eFast SAE Location" + str(time_step) + "us", cornerLocation=allEFastQueue)

                SAELocation = np.ones((HEIGHT, WIDTH), dtype=np.uint8)
                SAELocation *= 128
                for j, (img, locX, locY, recEventNo, p) in enumerate(allArcStarQueue):
                    if p:
                        SAELocation[locY][locX] = 255 # for On Events
                    else:
                        SAELocation[locY][locX] = 0 # for Off Events
                drawEventLocation(SAELocation, i, eventSpecified, name="All ArcStar SAE Location" + str(time_step) + "us", cornerLocation=allArcStarQueue)

                break

            last_event = packet['events']['t'][i]
            eventCount += 1

            if i == 0:
                accumulated_time = 0
            else:
                accumulated_time += (packet['events']['t'][i] - packet['events']['t'][i-1])
            
            if accumulated_time >= time_step:
                frame += 1
                accumulated_time = 0

if __name__ == "__main__":
    if not os.path.isdir("../Output"):
        os.mkdir("../Output")

    read_aedat4(aedat4_file, time_step=33000, processAll=False, eventSpecified=True, eventNo=102342) # 33000ms interval, @event=54000, we get arc* response, no for eFast
    # 5000ms interval, @eventNo 44000, eFast response, no arc* response
    # 67239
