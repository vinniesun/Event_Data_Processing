import pandas as pd
import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, dataloader
from torchvision import transforms, utils
from tqdm import tqdm
# For decoding AEDAT files
import struct
import h5py

# Set Path to the annotation files, recordings, HDF5
annotation_path = "../DAVIS_Events/Annotation/"
recording_path = "../DAVIS_Events/Recording/"

# Store the names of the files
list_of_annotation = os.listdir(annotation_path)
list_of_recording = os.listdir(recording_path)

########################################################################################################################################
#
# Use Pandas to read annotation file to make sure it's working as intended
#
########################################################################################################################################
#df = pd.read_csv(annotation_path + list_of_annotation[0], 
#                names=['Time(us)', 'x-Location', 'y-Location', 'x-size', 'y-size', 'track-num', 'class'],
#                comment='#', index_col=False)
#print(list_of_annotation[0])
#for col in df.columns:
#    print(col)
#print(df.head())

########################################################################################################################################
#
# read annotation data
#
########################################################################################################################################
def read_annotation(filename: str) -> np.ndarray:
    df = pd.read_csv(filename, 
                names=['Time(us)', 'x-Location', 'y-Location', 'x-size', 'y-size', 'track-num', 'class'],
                comment='#', index_col=False)
    
    return df.columns, df.to_numpy()

########################################################################################################################################
#
# Remove all the zeros
#
########################################################################################################################################
def remove_by_indices(TD, indices) -> dict:
    #
    return TD

########################################################################################################################################
#
# read .bin data
#
########################################################################################################################################
def read_dvs_data(filename: str, columns: list) -> dict:
    raw_data = open(filename, 'rb')
    
    # Check version number?
    temp = raw_data.readline().strip(b"\n")
    
    if temp[0] == ord('#'):
        version = 0
    elif temp[0] == ord('v'):
        version = temp[1:].decode()
    else:
        print("file is: ", str(temp, 'utf-8'))

    # Skip the rest of the comments
    original_pos = raw_data.tell()
    while True:
        temp = raw_data.readline().strip(b"\n")

        if len(temp) == 0:
            original_pos = raw_data.tell()
            break
        elif temp[0] == ord('#'):
            original_pos = raw_data.tell()
        else:
            break
    
    raw_data.seek(original_pos, 0)

    if version == 0:
        resolution = [304, 240]
    else:
        resolution = raw_data.read(4)
        resolution = [r for r in resolution if r != 0]
        _ = raw_data.readline().strip(b"\n")
    original_pos = raw_data.tell()

    print(resolution) # resolution = [width, height]

    raw_data_buffer = raw_data.read()
    rdb_len = len(raw_data_buffer)
    raw_data_buffer = np.array(struct.unpack('{}B'.format(rdb_len), raw_data_buffer), dtype=np.ubyte)
    #print(len(raw_data_buffer_old))
    #raw_data_buffer = np.empty(0, dtype=np.ubyte)
    #temp = raw_data.read(1)
    #while temp != b"":
    #    raw_data_buffer += np.array(struct.unpack('B', temp), dtype=np.ubyte) # read the rest of the data
    #    temp = raw_data.read(1)

    # Create TD (temporal difference) structure
    total_events = len(raw_data_buffer)
    TD = {}
    TD['x'] = np.zeros(total_events, dtype=np.ushort)
    TD['y'] = np.zeros(total_events, dtype=np.ushort)
    TD['p'] = np.zeros(total_events, dtype=np.ubyte)
    TD['ts'] = np.zeros(total_events, dtype=np.uintc)
    TD['type'] = np.ones(total_events, dtype=np.ubyte)*float('inf')

    # Read the buffer, one packet at a time, till the end of the buffer
    total_events = 0
    buffer_location = 0
    while buffer_location < len(raw_data_buffer):
        num_events = (np.array(raw_data_buffer[buffer_location+3], dtype=np.uintc) << 24) + (np.array(raw_data_buffer[buffer_location+2], dtype=np.uintc) << 16) + (np.array(raw_data_buffer[buffer_location+1], dtype=np.uintc) << 8) + np.array(raw_data_buffer[buffer_location], dtype=np.uintc)
        buffer_location += 4
        start_time = (np.array(raw_data_buffer[buffer_location+3], dtype=np.uintc) << 24) + (np.array(raw_data_buffer[buffer_location+2], dtype=np.uintc) << 16) + (np.array(raw_data_buffer[buffer_location+1], dtype=np.uintc) << 8) + np.array(raw_data_buffer[buffer_location], dtype=np.uintc)
        
        if version != 0:
            start_time = start_time << 16
        
        buffer_location += 8 # Skip over the end time

        if len(raw_data_buffer) >= (buffer_location + 8*num_events-1):
            current_type = np.array((raw_data_buffer[buffer_location:(buffer_location+(8*(num_events-1))):8]), dtype=np.ubyte) #[start:end:increments]
            current_subtype = np.array((raw_data_buffer[buffer_location+1:(buffer_location+8*(num_events)):8]), dtype=np.ubyte)
            y = (np.array(raw_data_buffer[(buffer_location+2):(buffer_location+8*(num_events)+1):8], dtype=np.ushort) + 256*np.array(raw_data_buffer[(buffer_location+3):(buffer_location+8*(num_events)+1):8], dtype=np.ushort))
            x = ((np.array(raw_data_buffer[(buffer_location+5):(buffer_location+8*(num_events)+4):8], dtype=np.ushort) << 8) + np.array(raw_data_buffer[(buffer_location+4):(buffer_location+8*(num_events)+3):8], dtype=np.ushort))
            ts = ((np.array(raw_data_buffer[(buffer_location+7):(buffer_location+8*(num_events)+6):8], dtype=np.ushort) << 8) + np.array(raw_data_buffer[(buffer_location+6):(buffer_location+8*(num_events)+5):8], dtype=np.ushort))

            buffer_location += num_events*8
            ts += start_time

            if version == 0:
                overflows = np.argwhere(current_type == 2)
                for i in range(1, len(overflows)):
                    ts[overflows[i]:] = ts[overflows[i]:] + 65536
            
            for i, j in zip(range(total_events, (total_events+num_events-1)), range(len(current_type))):
                TD['type'][i] = current_type[j]
                TD['x'][i] = x[j]
                TD['y'][i] = y[j]
                TD['p'][i] = current_subtype[j]
                TD['ts'][i] = ts[j]

            total_events += num_events
        else:
            buffer_location = len(raw_data_buffer)

    raw_data.close()
    del raw_data_buffer

    # Sort by ts?
    TD['x']

    return TD

########################################################################################################################################
#
# Explore data
#
########################################################################################################################################
# Loop through all of the files
#for i in tqdm(range(len(list_of_recording)), desc='file number'):
for i in tqdm(range(1), desc='file number'):
    filename = list_of_recording[i]

    # Load the annotation of current recording
    columns, df = read_annotation(annotation_path + list_of_annotation[i])
    
    temporal_data = read_dvs_data(recording_path + filename, columns)
    print(temporal_data['ts'])
