"""
    Generate the training images and text file needed for training a NN model
"""
import os
from tqdm import tqdm
import cv2

from process_events import *


annotation_file = "/Users/vincent/Desktop/CityUHK/EBBINNOT/EBBINNOT_AEDAT4/Annotation/20180711_Site1_3pm_12mm_01.txt"
event_file = "/Users/vincent/Desktop/CityUHK/EBBINNOT/EBBINNOT_AEDAT4/Recording/20180711_Site1_3pm_12mm_01.txt"

output_folder = "../Output"

def read_annotation(filename):
    print("read")

def imageChannel(image):
	dummy_channel = np.zeros((98061, 42, 42, 1))
	print(dummy_channel.shape)
	image = np.append(image, dummy_channel, axis=3)
	print(image.shape)
	
	return image

def saveImages(images):
    folder = "images/"
    location = DATA_PATH + folder
    if not os.path.isdir(location):
        os.mkdir(location)
    
    for i, im in enumerate(images):
        cv2.imwrite(location + str(i) + ".jpg", 255*im)

def trainLabels(labels):
    train_size = int(98061*0.8)

    trainName = "2007_train.txt"
    valName = "2007_val.txt"

    for i in tqdm(range(train_size), desc="processing training file"):
        with open(trainName, 'a') as f:
            f.write("/home/vincesun/YoloV4-Tiny/yolov4-tiny-pytorch/VOCdevkit/VOC2007/JPEGImages/" + str(i) + ".jpg" + " 187,187,229,229," + str(labels[i]) + "\n")
    
    for i in tqdm(range(train_size, 98061), desc="processing validation file"):
        with open(valName, 'a') as f:
            f.write("/home/vincesun/YoloV4-Tiny/yolov4-tiny-pytorch/VOCdevkit/VOC2007/JPEGImages/" + str(i) + ".jpg" + " 187,187,229,229," + str(labels[i]) + "\n")

def main():
    # Check output folder is there, if not, create it
    if not os.path.isdir(output_folder):
        os.mkdir(output_folder)

    # Read the event file and save all of the event
    currentEvent = aedat_to_events(event_file)

    # Apply refractory filtering
    #currentEvent.events, currentEvent.num_events = refractory_filtering(currentEvent, 2000)

    # Split all of the events and generate frames, @66ms
    ...


if __name__ == "__main__":
    main()