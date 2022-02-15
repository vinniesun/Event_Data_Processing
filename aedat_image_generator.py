from math import ceil
import aedat
import numpy as np

import cv2
import os
import glob
import pandas as pd
from tqdm import tqdm

########################################################################################################################################
#
# Global Parameters
#
########################################################################################################################################
aedat4_file = '../EBBINNOT_AEDAT4/Recording/*.aedat4'
image_path = '../images/'
mode = ['1b1c/', '1b2c/']
WIDTH = 240
HEIGHT = 180
CHANNEL = 3
#time_step = 66000
dt = 1
frames = 0
time = 0

def make_dir(mode: str) -> None:
    location = image_path+mode
    if not os.path.isdir(location):
        os.mkdir(location)

########################################################################################################################################
#
# Reading Annotation Files
#
########################################################################################################################################
def read_annotation(filename: str) -> np.ndarray:
    df = pd.read_csv(filename, 
                names=['Time(us)', 'x-Location', 'y-Location', 'x-size', 'y-size', 'track-num', 'class'],
                comment='#', index_col=False)
    
    return df.columns, df

def read_pickle(filename: str):
    return pd.read_pickle(filename)

########################################################################################################################################
#
# Reading the AEDAT4 Files and outputting image with 1b1c (1 bit 1 channel) format
#
# packet contains the follwowing fields:
#     - 'events'
#     - 'frame'
#     - 'imus'
#     - 'triggers'
#
#     In 'events', we have the following:
#     - 't', current time
#     - 'x', the width pixel
#     - 'y', the height pixel
#     - 'on', a boolean
# here, each "frame" is based on event's occuring over the last x ms. X is called time_step in this script
#
########################################################################################################################################
def read_aedat4_1b1c(filename: str, time_step=66000, processAll=True, factor=1) -> np.ndarray:
    data = aedat.Decoder(filename)
    for packet in data:
        total_events = len(packet['events'])
        frames = ceil(packet['events']['t'][-1]/time_step)
        image = np.zeros((frames, HEIGHT, WIDTH), dtype=np.ubyte)
        #image = np.zeros((frames, HEIGHT, WIDTH), dtype=np.uint32)

        frame = 0
        accumulated_time = 0
        for i in tqdm(range(0, total_events, dt), desc='processing events'):
            y = packet['events']['y'][i]
            x = packet['events']['x'][i]
            image[frame][y][x] = 255

            if i == 0:
                accumulated_time = 0
            else:
                accumulated_time += (packet['events']['t'][i] - packet['events']['t'][i-1])
            
            if accumulated_time >= time_step:
                frame += 1
                accumulated_time = 0

    return image

########################################################################################################################################
#
# Reading the AEDAT4 Files and outputting image with 1b2c (1 bit 2 channel) format
# Channel 0: ON spike channel
# Channel 1: OFF Spike channel
# Channel 2: N/A (exists for RGB syntax)
#
# packet contains the follwowing fields:
#     - 'events'
#     - 'frame'
#     - 'imus'
#     - 'triggers'
#
#     In 'events', we have the following:
#     - 't', current time
#     - 'x', the width pixel
#     - 'y', the height pixel
#     - 'on', a boolean
# here, each "frame" is based on event's occuring over the last x ms. X is called time_step in this script
#
########################################################################################################################################
def read_aedat4_1b2c(filename: str, time_step=66000, processAll=True, factor=1) -> np.ndarray:
    data = aedat.Decoder(filename)
    for packet in data:
        total_events = len(packet['events'])
        frames = ceil(packet['events']['t'][-1]/time_step)
        image = np.zeros((frames, HEIGHT, WIDTH, CHANNEL), dtype=np.ubyte)

        frame = 0
        accumulated_time = 0
        for i in tqdm(range(0, total_events, dt), desc='processing events'):
            y = packet['events']['y'][i]
            x = packet['events']['x'][i]
            if packet['events']['on'][i]:
                image[frame][y][x][0] = 255     # ON Channel
            else:
                image[frame][y][x][1] = 255     # OFF Channel

            if i == 0:
                accumulated_time = 0
            else:
                accumulated_time += (packet['events']['t'][i] - packet['events']['t'][i-1])
            
            if accumulated_time >= time_step:
                frame += 1
                accumulated_time = 0

    return image

########################################################################################################################################
#
# Save Processed Event's as images
#
########################################################################################################################################
def save_image(mode: str, folder_name: str, image: np.ndarray) -> None:
    location = image_path+mode+folder_name
    if not os.path.isdir(location):
        os.mkdir(location)

    # Store the images from image np array
    #img_no = 0
    for i, img in tqdm(enumerate(image)):
        cv2.imwrite(location + str(i) + '.jpg', img)
        #cv2.imwrite(location + str(img_no) + '.jpg', img)
        #img_no += 1

########################################################################################################################################
#
# Save Processed Event's as video
#
########################################################################################################################################
def save_video(mode: str, folder_name: str, filename: str) -> None:
    # Store the output as a video
    location = image_path+mode+folder_name
    img_array = []
    for f in glob.glob(location+'*.jpg'):
        img = cv2.imread(f)
        height, width, _ = img.shape
        size = (width, height)
        img_array.append(img)
    
    vid_recorder = cv2.VideoWriter(location+filename+'.avi', cv2.VideoWriter_fourcc(*'DIVX'), 15, size)

    for i in img_array:
        vid_recorder.write(i)
    vid_recorder.release()

########################################################################################################################################
#
# Display processed events
#
########################################################################################################################################
def display_video(mode: str, folder_name: str) -> None:
    i = 0
    location = image_path+mode+folder_name
    files = glob.glob(location+'*.jpg')
    flag = True
    while flag:
        for f in sorted(files):
            img = cv2.imread(f)
            cv2.imshow('frame', img)

            if cv2.waitKey(60) & 0xFF == ord('q'):
                flag = False
                break

    cv2.destroyAllWindows()

if __name__ == '__main__':
    
    recordings = glob.glob(aedat4_file)

    for m in mode:
        make_dir(m)
    
    for i, r in tqdm(enumerate(recordings), desc='recordings'):
        filename = r.split('/')[-1]
        #current_events = read_aedat4_1b1c(r)
        current_events = read_aedat4_1b2c(r)
        folder_name = filename.split('.')[0] + '/'
        save_image(mode[1], folder_name, current_events)
        save_video(mode[1], folder_name, filename.split('.')[0])
        #display_video(mode[0], folder_name)
    
    #df = read_pickle("../GT_ON_OFF_data_w_augmentation/GT_train_42x42_20190924-meta-data.pkl")
    #print(df.head(20))
