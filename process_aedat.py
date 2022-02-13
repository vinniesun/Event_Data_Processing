import pandas as pd
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, dataloader
from torchvision import transforms, utils
# For decoding AEDAT files
import struct
import h5py

# Set Path to the annotation files, recordings, HDF5
annotation_path = "../EBBINNOT_AEDAT4/Annotation/"
recording_path = "../EBBINNOT_AEDAT4/Recording/"
HDF5_path = "../EBBINNOT_AEDAT4/HDF5/"

# Store the names of the files
list_of_annotation = os.listdir(annotation_path)
list_of_recording = os.listdir(recording_path)

########################################################################################################################################
#
# Use Pandas to read annotation file to make sure it's working as intended
#
########################################################################################################################################
df = pd.read_csv(annotation_path + list_of_annotation[0], 
                names=['Time(us)', 'x-Location', 'y-Location', 'x-size', 'y-size', 'track-num', 'class'],
                comment='#', index_col=False)
#print(list_of_annotation[0])
#for col in df.columns:
#    print(col)
#print(df.head())

########################################################################################################################################
#
# Read an AEDAT4 file
#
########################################################################################################################################
#def gather_aedat(directory, start_id, end_id, filename_prefix = 'user'):
def gather_aedat(directory, filename):
    """
        glob is a Python package used to find all the pathnames that matches a specified pattern according
        to Unix shell.
    """
    import glob
    fns = []
    """
    for i in range(start_id,end_id):
        #search_mask = directory+'/'+filename_prefix+"{0:02d}".format(i)+'*.aedat'
        glob_out = glob.glob(search_mask)
        if len(glob_out)>0:
            fns+=glob_out
    """
    search_mask = directory + filename
    glob_out = glob.glob(search_mask)
    if len(glob_out) > 0:
        fns += glob_out
    return fns
"""
    This method breaks down how AEDAT files are read.
    param:
    filename: name of the AEDAT file
    labels_name: label's name
"""
def aedat_to_events(filename, labels_name, dataframe):
    label_filename = labels_name
    #labels = np.loadtxt(label_filename, skiprows=1, delimiter=',',dtype='uint32')
    #labels = np.loadtxt(label_filename, delimiter=',')
    labels = dataframe.to_numpy()
    events=[]
    with open(filename, 'rb') as f:
        for i in range(5):
            f.readline()
        while True: 
            data_ev_head = f.read(28) # Read 28 Bytes at once
            if len(data_ev_head)==0: break

            """
            Common AEDAT Header contains the following:

            Bytes   |   Meaning         |   Description
            --------------------------------------------------------------------------------------------------------------------------------|
            0~1     |   eventType       |   numerical type ID, unique to each event type                                                    |
            --------------------------------------------------------------------------------------------------------------------------------|
            2~3     |   eventSource     |   numerical source ID, identifies who generated the events inside a system                        |
            --------------------------------------------------------------------------------------------------------------------------------|
            4~7     |   eventSize       |   Size of one event in bytes                                                                      |
            --------------------------------------------------------------------------------------------------------------------------------|
            8~11    |   eventTSOffset   |   Offset from the start of an event in bytes, at which the main 32 bit time-stamp can be found    |
            --------------------------------------------------------------------------------------------------------------------------------|
            12~15   |   eventTSOverflow |   overflow counter for the standard 32bit event time-stamp. Used to generate the 64bit time-stamp |
            --------------------------------------------------------------------------------------------------------------------------------|
            16~19   |   eventCapacity   |   Max number of events this packet can store                                                      |
            --------------------------------------------------------------------------------------------------------------------------------|
            20~23   |   eventNumber     |   Total number of events present in this packet (valid and invalid)                               |
            --------------------------------------------------------------------------------------------------------------------------------|
            24~27   |   eventValid      |   Total number of valid events present in this packet                                             |
            --------------------------------------------------------------------------------------------------------------------------------|
            """
            """
            For Python's struct.unpack(), the format string are:
            'H': unsigned short (2 Bytes)
            'I': Integer (4 Bytes)
            """
            eventtype = struct.unpack('H', data_ev_head[0:2])[0]
            eventsource = struct.unpack('H', data_ev_head[2:4])[0]
            eventsize = struct.unpack('I', data_ev_head[4:8])[0]
            eventoffset = struct.unpack('I', data_ev_head[8:12])[0]
            eventtsoverflow = struct.unpack('I', data_ev_head[12:16])[0]
            eventcapacity = struct.unpack('I', data_ev_head[16:20])[0]
            eventnumber = struct.unpack('I', data_ev_head[20:24])[0]
            eventvalid = struct.unpack('I', data_ev_head[24:28])[0]

            print(eventtype, eventnumber, eventsize)

            if(eventtype == 1):
                event_bytes = np.frombuffer(f.read(eventnumber*eventsize), 'uint32')
                event_bytes = event_bytes.reshape(-1,2)

                x = (event_bytes[:,0] >> 17) & 0x00001FFF
                y = (event_bytes[:,0] >> 2 ) & 0x00001FFF
                p = (event_bytes[:,0] >> 1 ) & 0x00000001
                t = event_bytes[:,1]
                events.append([t,x,y,p])

            else:
                f.read(eventnumber*eventsize)
    events = np.column_stack(events)
    events = events.astype('uint32')
    clipped_events = np.zeros([4,0],'uint32')
    for l in labels:
        """
        start = np.searchsorted(events[0,:], l[1])
        end = np.searchsorted(events[0,:], l[2])
        """
        clipped_events = np.column_stack([clipped_events,events[:,start:end]])
    return clipped_events.T, labels

"""
    create HDF5 file that contains the event of the aedat file
    Input param:
    directory: the directory that holds the aedat files
    filename: the name of the file that will be stored as a HDF5
    labelname: the name of the annotation file that contains the label of "filename"
"""
def create_events_hdf5(directory, filename, labelname):
    #fns_train = gather_aedat('/share/data/DvsGesture/aedat/',1,24)
    #fns_test = gather_aedat('/share/data/DvsGesture/aedat/',24,30)
    event_data = gather_aedat(directory, filename)

    with h5py.File('data/dvs_gestures_events.hdf5', 'w') as f:
        f.clear()

        print("processing training data...")
        key = 0
        train_grp = f.create_group('Event_Data')
        for file_d in event_data:
            print(key)
            events, labels = aedat_to_events(file_d)
            subgrp = train_grp.create_group(str(key))
            dset_dt = subgrp.create_dataset('time', events[:,0].shape, dtype=np.uint32)
            dset_da = subgrp.create_dataset('data', events[:,1:].shape, dtype=np.uint8)
            dset_dt[...] = events[:,0]
            dset_da[...] = events[:,1:]
            dset_l = subgrp.create_dataset('labels', labels.shape, dtype=np.uint32)
            dset_l[...] = labels
            key += 1

        """
        print("processing testing data...")
        key = 0
        test_grp = f.create_group('test')
        for file_d in fns_test:
            print(key)
            events, labels = aedat_to_events(file_d)
            subgrp = test_grp.create_group(str(key))
            dset_dt = subgrp.create_dataset('time', events[:,0].shape, dtype=np.uint32)
            dset_da = subgrp.create_dataset('data', events[:,1:].shape, dtype=np.uint8)
            dset_dt[...] = events[:,0]
            dset_da[...] = events[:,1:]
            dset_l = subgrp.create_dataset('labels', labels.shape, dtype=np.uint32)
            dset_l[...] = labels
            key += 1
        """

        #stats =  gather_gestures_stats(train_grp) Don't need this i think?
        #f.create_dataset('stats',stats.shape, dtype = stats.dtype)
        #f['stats'][:] = stats

def processAEDAT(HDF5_path):
    # Check the folder for HDF5 is created
    if not os.path.isdir(HDF5_path):
        os.mkdir(HDF5_path)

    # Begin Processing

event, labels = aedat_to_events(recording_path+list_of_recording[0], annotation_path+list_of_annotation[0], df)