import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys

from torch.utils.data import Dataset, TensorDataset, DataLoader
from torchvision import transforms

import gzip
import cv2

from tqdm import tqdm

BATCH_SIZE = 128
DATA_PATH = "../Train"
IMAGE_PATH = "../images/1b1c/"
META_DATA_PATH = "../GT_ON_OFF_data_w_augmentation/GT_train_42x42_20190924-meta-data.pkl"
FIXED_SIZE = 42
COLUMN_HEADER = ["aumented", "bbox", "boundaryPt", "classID", "fileID", "fileName", "scene_und_vect", "track_id", "vel_dir", "xSize", "ySize"]
NAME_MAP = {"ENG": "Site1", "LT4": "Site2", "PGP": "Site3"}

def openFiles(image_path, label_path, meta_data_path):
    if image_path:
        with gzip.open(label_path, 'rb') as lbpath:
            labels = np.frombuffer(lbpath.read(), dtype=np.uint8, offset=8)
    else:
        labels = None
    
    if label_path:
        with gzip.open(image_path, 'rb') as imgpath:
            images = np.frombuffer(imgpath.read(), dtype=np.uint8, offset=16).reshape(len(labels), FIXED_SIZE, FIXED_SIZE, -1) # This is to evaluate the image as a 2-D array, ie image
    else:
        images = None

    if meta_data_path:
        meta_data = pd.read_pickle(meta_data_path)
    else:
        meta_data = None

    return images, labels, meta_data

def processMetaData(df):
    print(df.columns)
    df_arr = df.to_numpy()
    print(df_arr[:5])

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
    

def saveLabels(labels):
    folder = "labels/"
    location = DATA_PATH + folder
    if not os.path.isdir(location):
        os.mkdir(location)

    fileName = location + "labels.txt"
    with open(fileName, 'a') as f:
        for l in labels:
            f.write(str(l) + "\n")

transform = transforms.Compose([
    transforms.ToTensor()
])

class customDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        x = self.images[index]
        y = self.labels[index]

        if self.transform:
            x = self.transform(x)

        return x, y

def drawSamples(train_loader):
    samples, labels = iter(train_loader).next()
    labels = [l.item() for l in labels[:32]]
    plt.figure(figsize=(16,16))
    for i, sample in enumerate(samples[:32]):
        plt.subplot(4, 8, i+1)
        plt.title(labels[i])
        plt.axis('off')
        plt.imshow(np.transpose(sample, (1, 2, 0)))
    plt.show()

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

if __name__ == '__main__':
    images, labels, meta_data = openFiles("", "", META_DATA_PATH)
    processMetaData(meta_data)
    #images = imageChannel(images)
    #image_dataset = customDataset(images, labels, transform)
    #train_loader = DataLoader(image_dataset, batch_size=32, shuffle=True, drop_last=True)
    #drawSamples(train_loader)
    #saveImages(images)
    #saveLabels(labels)
    #trainLabels(labels)
