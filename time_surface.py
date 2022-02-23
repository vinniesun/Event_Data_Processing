from math import ceil, sqrt
import aedat
import os
from tqdm import tqdm
from efast import *
from arcstar import *
from util import *
from harris_detector import *
from plot_tools import *

aedat4_file = '../EBBINNOT_AEDAT4/Recording/20180711_Site1_3pm_12mm_01.aedat4'
dt = 1

def read_aedat4(filename:str, time_step:int=66000, processAll:bool=True, factor:int=1, eventSpecified:bool=False, eventNo:int=None) -> None:
    data = aedat.Decoder(filename)
    for packet in data:
        total_events = len(packet['events'])
        noFrames = ceil(ceil(packet['events']['t'][-1]/time_step))
        sae = np.zeros((HEIGHT, WIDTH, 2), dtype=np.int64)
        ebbi_image = np.zeros((noFrames, HEIGHT, WIDTH, 2), dtype=np.uint8)

        last_event = 0
        
        pastOnEvents = []
        pastOffEvents = []

        threshold_val = time_step

        frame = 0
        accumulated_time = 0

        for i in tqdm(range(0, total_events, dt), desc='processing events'):
            y = packet['events']['y'][i]
            x = packet['events']['x'][i]
            pol = packet['events']['on'][i] # packet['events']['on'] = 1 means it's ON, packet['event']['on'] = 0 means it's OFF

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
            if eventSpecified and i == eventNo:
            #if isCornerEFast(image, x, y, int(pol)) and not isCornerArcStar(image, prev_state, prev_state_inv, x, y, int(pol)) and i >= 5000:
                #filtered = medianFilter(image, 3)
                #image = filtered

                # Draw the Current EBBI Image.
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

                """
                fig = plt.figure(figsize=(16, 16))
                plt.tight_layout()
                for j, (row, col) in enumerate(cornerLocation):
                    ax = fig.add_subplot(noRow, noCol, j+1, projection='3d')
                    temp = crop(image, col, row, 9)
                    ax = draw3DBarGraphSub(temp, i, subplot=ax, title="Corner No. " + str(j))
                plt.savefig("../Output/" + str(time_step) + "us 3D Barplot of Harris Corner of Event " + str(i) + ".jpg", dpi=300)
                plt.show()
                plt.close(fig)
                """
                
                """
                # Plot eFast Response
                fig, ax = plt.subplots(1, 1,figsize=(12, 9))
                plt.tight_layout()
                if isCornerEFast(image, x, y, int(pol)):
                    temp = crop(image, x, y, 9)
                    ax = drawHeatMapSub(temp, i, subplot=ax, title="Corner")
                plt.savefig("../Output/" + str(time_step) + "us 2D eFast Corner of Event " + str(i) + ".jpg", dpi=300)
                plt.show()
                plt.close(fig)

                fig = plt.figure(figsize=(12, 9))
                plt.tight_layout()
                ax = fig.add_subplot(1, 1, 1, projection='3d')
                if isCornerEFast(image, x, y, int(pol)):
                    temp = crop(image, x, y, 9)
                    ax = draw3DBarGraphSub(temp, i, subplot=ax, title="Corner")
                plt.savefig("../Output/" + str(time_step) + "us 3D Barplot of eFast Corner of Event " + str(i) + ".jpg", dpi=300)
                plt.show()
                plt.close(fig)
                """

                # Plot All eFast Response
                ecornerCount = 0
                ecornerLocation = []
                for r in range(len(image)):
                    for c in range(len(image[0])):
                        if isCornerEFast(image, c, r, int(pol)):
                            ecornerCount += 1
                            ecornerLocation.append((r, c)) # store as row, col pair
                if ecornerCount > 0:
                    # Draw SAE with corner highlighted
                    temp = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
                    for row, col in ecornerLocation:
                        temp[row][col] = 255

                    drawHeatMapWhole(temp, i, eventSpecified, name="SAE Location of eFast Corners" + str(time_step) + "us")

                    noRow, noCol = int((sqrt(ecornerCount))), ceil(ecornerCount / int(sqrt(ecornerCount)))

                    fig, ax = plt.subplots(noRow, noCol,figsize=(30, 30))
                    plt.tight_layout()
                    for j, (row, col) in enumerate(ecornerLocation):
                        temp = crop(image, col, row, 9)
                        #ax[j // noCol][j % noCol] = drawHeatMapSub(temp, i, subplot=ax[j // noCol][j % noCol], title="Corner " + str(row) + ", " + str(col))
                        ax[j // noCol][j % noCol] = drawHeatMapSub(temp, i, subplot=ax[j // noCol][j % noCol])
                    plt.savefig("../Output/" + str(time_step) + "us 2D eFast, Corners detected - " + str(ecornerCount) + ", of Event " + str(eventNo) + ".jpg", dpi=100)
                    plt.show()
                    plt.close(fig)
                
                # Plot All ArcStar Response
                acornerCount = 0
                acornerLocation = []
                for r in range(len(image)):
                    for c in range(len(image[0])):
                        if isCornerArcStar(image, prev_state, prev_state_inv, c, r, int(pol)):
                            acornerCount += 1
                            acornerLocation.append((r, c)) # store as row, col pair
                if acornerCount > 0:
                    temp = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
                    for row, col in acornerLocation:
                        temp[row][col] = 255

                    drawHeatMapWhole(temp, i, eventSpecified, name="SAE Location of ArcStars Corners" + str(time_step) + "us")

                    noRow, noCol = int((sqrt(acornerCount))), ceil(acornerCount / int(sqrt(acornerCount)))

                    fig, ax = plt.subplots(noRow, noCol,figsize=(30, 30))
                    plt.tight_layout()
                    for j, (row, col) in enumerate(acornerLocation):
                        temp = crop(image, col, row, 9)
                        #ax[j // noCol][j % noCol] = drawHeatMapSub(temp, i, subplot=ax[j // noCol][j % noCol], title="Corner " + str(row) + ", " + str(col))
                        ax[j // noCol][j % noCol] = drawHeatMapSub(temp, i, subplot=ax[j // noCol][j % noCol])
                    plt.savefig("../Output/" + str(time_step) + "us 2D ArcStar, Corners detected - " + str(acornerCount) + ", of Event " + str(eventNo) + ".jpg", dpi=100)
                    plt.show()
                    plt.close(fig)

                # Generate only eFast Location, only AStar Location and both
                onlyEFast, onlyAStar, both = [], [], []
                for row, col in ecornerLocation:
                    if (row, col) not in acornerLocation:
                        onlyEFast.append((row, col))
                    else:
                        both.append((row, col))

                for row, col in acornerLocation:
                    if (row, col) not in both:
                        onlyAStar.append((row, col))

                # Plot Only eFast
                temp = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
                for row, col in onlyEFast:
                    temp[row][col] = 255

                drawHeatMapWhole(temp, i, eventSpecified, name="SAE Location of only EFast Corners" + str(time_step) + "us")

                noRow, noCol = int((sqrt(len(onlyEFast)))), ceil(len(onlyEFast) / int(sqrt(len(onlyEFast))))
                fig, ax = plt.subplots(noRow, noCol,figsize=(30, 30))
                plt.tight_layout()
                for j, (row, col) in enumerate(onlyEFast):
                    temp = crop(image, col, row, 9)
                    ax[j // noCol][j % noCol] = drawHeatMapSub(temp, i, subplot=ax[j // noCol][j % noCol], title=str(row) + ", " + str(col))
                plt.savefig("../Output/" + str(time_step) + "us 2D only eFast, Corners detected - " + str(len(onlyEFast)) + ", of Event " + str(eventNo) + ".jpg", dpi=100)
                plt.show()
                plt.close(fig)

                fig = plt.figure(figsize=(16, 16))
                plt.tight_layout()
                for j, (row, col) in enumerate(onlyEFast):
                    ax = fig.add_subplot(noRow, noCol, j+1, projection='3d')
                    temp = crop(image, col, row, 9)
                    ax = draw3DBarGraphSub(temp, i, subplot=ax, title=str(row) + ", " + str(col))
                plt.savefig("../Output/" + str(time_step) + "us 3D Barplot of only eFast Corner of Event " + str(i) + ".jpg", dpi=100)
                plt.show()
                plt.close(fig)

                # Plot Only Arc*
                temp = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
                for row, col in onlyAStar:
                    temp[row][col] = 255

                drawHeatMapWhole(temp, i, eventSpecified, name="SAE Location of only ArcStar Corners" + str(time_step) + "us")

                noRow, noCol = int((sqrt(len(onlyAStar)))), ceil(len(onlyAStar) / int(sqrt(len(onlyAStar))))
                fig, ax = plt.subplots(noRow, noCol,figsize=(30, 30))
                plt.tight_layout()
                for j, (row, col) in enumerate(onlyAStar):
                    temp = crop(image, col, row, 9)
                    ax[j // noCol][j % noCol] = drawHeatMapSub(temp, i, subplot=ax[j // noCol][j % noCol], title=str(row) + ", " + str(col))
                plt.savefig("../Output/" + str(time_step) + "us 2D only AStar, Corners detected - " + str(len(onlyAStar)) + ", of Event " + str(eventNo) + ".jpg", dpi=100)
                plt.show()
                plt.close(fig)

                # Plot Both
                temp = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
                for row, col in both:
                    temp[row][col] = 255

                drawHeatMapWhole(temp, i, eventSpecified, name="SAE Location of Both Corners" + str(time_step) + "us")

                noRow, noCol = int((sqrt(len(both)))), ceil(len(both) / int(sqrt(len(both))))
                fig = plt.figure(figsize=(30, 30))
                plt.tight_layout()
                for j, (row, col) in enumerate(both):
                    ax = fig.add_subplot(noRow, noCol, j+1)
                    temp = crop(image, col, row, 9)
                    ax = drawHeatMapSub(temp, i, subplot=ax, title=str(row) + ", " + str(col))
                plt.savefig("../Output/" + str(time_step) + "us 2D Both, Corners detected - " + str(len(both)) + ", of Event " + str(eventNo) + ".jpg", dpi=100)
                plt.show()
                plt.close(fig)

                """
                #fig = plt.figure(figsize=(12, 9))
                #plt.tight_layout()
                #for j, (row, col) in enumerate(cornerLocation):
                #    ax = fig.add_subplot(noRow, noCol, j+1, projection='3d')
                #    if isCornerArcStar(image, prev_state, prev_state_inv, col, row, int(pol)):
                #        temp = crop(image, col, row, 9)
                #        ax = draw3DBarGraphSub(temp, i, subplot=ax, title="Corner No. " + str(j))
                #plt.savefig("../Output/" + str(time_step) + "us 3D Barplot of ArcStar Corner of Event " + str(eventNo) + ".jpg", dpi=300)
                #plt.show()
                #plt.close(fig)
                """

                """
                # Plot ArcStar Response
                fig, ax = plt.subplots(1, 1,figsize=(12, 9))
                plt.tight_layout()
                if isCornerArcStar(image, prev_state, prev_state_inv, x, y, int(pol)):
                    temp = crop(image, x, y, 9)
                    ax = drawHeatMapSub(temp, i, subplot=ax, title="Corner")
                plt.savefig("../Output/" + str(time_step) + "us 2D ArcStar Corner of Event " + str(i) + ".jpg", dpi=300)
                plt.show()
                plt.close(fig)

                fig = plt.figure(figsize=(12, 9))
                plt.tight_layout()
                ax = fig.add_subplot(1, 1, 1, projection='3d')
                if isCornerArcStar(image, prev_state, prev_state_inv, x, y, int(pol)):
                    temp = crop(image, x, y, 9)
                    ax = draw3DBarGraphSub(temp, i, subplot=ax, title="Corner")
                plt.savefig("../Output/" + str(time_step) + "us 3D Barplot of ArcStar Corner of Event " + str(i) + ".jpg", dpi=300)
                plt.show()
                plt.close(fig)
                """

                break

            last_event = packet['events']['t'][i]

            if i == 0:
                accumulated_time = 0
            else:
                accumulated_time += (packet['events']['t'][i] - packet['events']['t'][i-1])
            
            if accumulated_time >= time_step:
                frame += 1
                accumulated_time = 0
                ebbi_image[frame] = ebbi_image[frame-1]

if __name__ == "__main__":
    if not os.path.isdir("../Output"):
        os.mkdir("../Output")

    read_aedat4(aedat4_file, time_step=5000, processAll=False, eventSpecified=True, eventNo=72342) # 33000ms interval, @event=54000, we get arc* response, no for eFast
    # 5000ms interval, @eventNo 44000, eFast response, no arc* response
