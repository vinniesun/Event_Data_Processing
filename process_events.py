import numpy as np

from tqdm import tqdm

# For decoding AEDAT files
# for reading AEDAT4 files, we need to use Google's flatbuffers to decode the file,
# as AEDAT4 uses FlatBuffers to serialise data.
import dv

# The height and width of our incoming aedat file's frame's width & height
WIDTH = 240
HEIGHT = 180

class Events(object):
    def __init__(self, num_events: int, width: int, height: int) -> np.ndarray:
        # events contains the following index:
        # t: the timestamp of the event. 64 bit signed integer
        # x: the x position of the event. 8 bit unsigned integer
        # y: the y position of the event. 8 bit unsigned integer
        # p: the polarity of the event. a boolean
        self.events = np.zeros((num_events), dtype=[("t", np.int64), ("x", np.uint16), ("y", np.uint16), ("p", np.bool_)])
        self.width = width
        self.height = height
        self.num_events = num_events

    def sort_by_time(self):
        return np.sort(self.events, order="t", kind="mergesort")

"""
    Apply Refractory Filtering to all of the stored events.
    If the timestamp of the current event is not greater than the refractory_period (comparing to the event pixel's last event time),
    then the event is treated as noise/redundant and is discarded

    @param
    event_array: an object containing all of the events stored as a numpy array
    refractory_period: time in us

    @return
    The updated event stream
"""
def refractory_filtering(event_array: Events, refractory_period: int) -> np.ndarray:
    # This is used to store if each event i event_array satisfy the refractory window
    # If it doesn't, it will be stored as false.
    check = np.ones(event_array.num_events, dtype=np.bool_)
    prev_time = np.zeros((event_array.height, event_array.width)) - refractory_period

    for i, e in tqdm(enumerate(event_array.events)):
        x, y = e["x"], e["y"]

        if e["t"] - prev_time[y][x] < refractory_period:
            check[i] = False
        else:
            prev_time[y][x] = e["t"]

    return event_array.events[check], np.count_nonzero(check == True)

"""
    Background Activity Filtering. Only events which are correlated with a neighbouring event within time_window (in us) will be
    allowed through the filter

    Adapted from https://github.com/gorchard/event-Python/blob/master/eventvision.py

    @param
"""
def background_activity_filter(event_array: Events, time_window=200):
    max_x, max_y = event_array.width - 1, event_array.height - 1
    t0 = np.ones((event_array.height, event_array.width)) - time_window - 1
    x_prev, y_prev, p_prev = 0, 0, 0
    valid_indices = np.ones(event_array.num_events, dtype=np.bool_)

    for i, e in tqdm(enumerate(event_array.events)):
        ts, x, y, p = e["t"], e["x"], e["y"], e["p"]
        
        if x_prev != x or y_prev != y or p_prev != p:
            t0[y][x] = -time_window
            min_x_sub = max(0, x-1)
            max_x_sub = min(max_x, x+1)
            min_y_sub = max(0, y-1)
            max_y_sub = min(max_y, y+1)

            t0_temp = t0[min_y_sub:(max_y_sub+1), min_x_sub:(max_x_sub + 1)]

            if min(ts - t0_temp.reshape(-1, 1)) > time_window:
                valid_indices[i] = 0

        t0[y][x], x_prev, y_prev = ts, x, y

    return event_array.events[valid_indices], np.count_nonzero(valid_indices == True)

"""
    Generate sliced events based on number of events

    @param
    event_array: an object containing all of the events stored as a numpy array
    event_count: number of events we want to slice as a frame
"""
def event_count_slice(event_array: Events, event_count: int=2000, drop_incomplete: bool=True) -> list:
    total_event = len(event_array.events)
    event_count = min(event_count, total_event)

    if drop_incomplete:
        num_slices = int(np.floor((total_event - event_count) / event_count) + 1)
    else:
        num_slices = int(np.ceil((total_event - event_count) / event_count) + 1)

    indices_start = np.arange(num_slices) * event_count
    indices_end = indices_start + event_count

    return [event_array.events[indices_start[i]:indices_end[i]] for i in range(num_slices)]

"""
    Generate sliced events based on time window

    Adapted from https://github.com/neuromorphs/tonic/blob/develop/tonic/functional/slicing.py

    @param
    event_array: an object containing all of the events stored as a numpy array
    time_window: amount of time we want to accumulate to generate a frame (based on timeflow of the event's timestamp)
"""
def time_window_slice(event_array: Events, time_window: int=33000, drop_incomplete: bool=True) -> list:
    timestamps = event_array.events["t"]

    if drop_incomplete:
        num_slices = int(np.floor(((timestamps[-1] - timestamps[0]) - time_window)/time_window) + 1)
    else:
        num_slices = int(np.ceil(((timestamps[-1] - timestamps[0]) - time_window)/time_window) + 1)

    window_start_time = np.arange(num_slices) * time_window  + timestamps[0]
    window_end_time = window_start_time + time_window
    indices_start = np.searchsorted(timestamps, window_start_time)
    indices_end = np.searchsorted(timestamps, window_end_time)

    return [event_array.events[indices_start[i]:indices_end[i]] for i in range(num_slices)]

"""
    Generate the frames based on the number of slices from slice by time or slice by events
"""
def accumulate_and_generate(sliced_event_array: list, width, height) -> np.ndarray:
    frames = np.zeros((len(sliced_event_array), height, width, 2), dtype=np.uint8)
    mask = np.zeros((height, width, 2))

    for i, e in tqdm(enumerate(sliced_event_array)):
        # is there a more performant way to do this?
        for _, x, y, p in e:
            frames[i][y][x][int(p)] = 255

    return frames

"""
    Generate the time surface of our event.
"""
def generate_time_surface(event_array: Events, time_threshold: int, event_start: int=0, event_end: int=-1):
    surface_height, surface_width = event_array.height, event_array.width
    time_surface = np.zeros((surface_height, surface_width, 2), dtype=np.int32)
    mask = np.zeros((surface_height, surface_width, 2), dtype=np.bool_)

    prev_event_time = 0

    for e in event_array.events[event_start:event_end]:
        deltaT = e["t"] - prev_event_time
        
        time_surface[:, :, int(e["p"])] -= deltaT
        time_surface[e["y"]][e["x"]][int(e["p"])] = time_threshold
        mask = np.where(time_surface < 0)

        time_surface[mask] = 0

        prev_event_time = e["t"]

    return time_surface

def update_time_surface(time_surface: np.array, time_threshold: int, t, x, y, p, prev_time):
    deltaT = t - prev_time

    time_surface[:, :, int(p)] -= deltaT
    time_surface[y][x][int(p)] = time_threshold
    mask = np.where(time_surface < 0)

    time_surface[mask] = 0

    return time_surface


"""
    This method reads in events stored as an AEDAT4 file
    This is a much faster method

    code adapted from https://gitlab.com/synsense/aermanager/-/blob/master/aermanager/parsers.py

    param:
    filename: name of the AEDAT file
"""
def aedat_to_events(filename):
    # Read in the aedat file and store as raw bytes (1 byte at a time)
    i = 0
    with dv.AedatFile(filename) as f:
        for packet in f.numpy_packet_iterator("events"):
            x = packet["x"]
            y = packet["y"]
            t = packet["timestamp"]
            p = packet["polarity"]

    num_events = len(x)
    events = Events(num_events, width=WIDTH, height=HEIGHT)

    events.events["x"] = x
    events.events["y"] = y
    events.events["t"] = t
    events.events["p"] = p

    return events

"""
    This method reads in events stored as a .bin file
    It is very slow. Need to find a way to improve its speed

    code adapted from https://github.com/gorchard/event-Python/blob/master/eventvision.py
"""
def bin_to_events(filename):
    with open(filename, "rb") as f:
        # Looping through the header
        header = f.readline()

        while header[0] == "#":
            header = f.readline()

        raw_data = np.fromfile(f, dtype=np.uint8)

    num_of_events = len(raw_data)
    processed_events = Events(num_of_events, width=WIDTH, height=HEIGHT)
    full_x = np.zeros(num_of_events)
    full_y = np.zeros(num_of_events)
    full_t = np.zeros(num_of_events)
    full_p = np.zeros(num_of_events)
    full_f = np.zeros(num_of_events)
    indices = np.zeros(num_of_events, dtype=np.bool_)

    buffer_location, start_index = 0, 0

    while buffer_location < len(raw_data):
        num_events = ((raw_data[buffer_location + 3].astype(np.uint32) << 24) + (raw_data[buffer_location + 2].astype(np.uint32) << 16) + (raw_data[buffer_location + 1].astype(np.uint32) << 8) + raw_data[buffer_location])
        buffer_location = buffer_location + 4
        start_time = ((raw_data[buffer_location + 3].astype(np.uint32) << 24) + (raw_data[buffer_location + 2].astype(np.uint32) << 16) + (raw_data[buffer_location + 1].astype(np.uint32) << 8) + raw_data[buffer_location])
        buffer_location = buffer_location + 8

        event_type = raw_data[buffer_location:(buffer_location + 8 * num_events):8]
        event_subtype = raw_data[(buffer_location + 1):(buffer_location + 8 * num_events + 1):8]
        y = raw_data[(buffer_location + 2):(buffer_location + 8 * num_events + 2):8]
        x = ((raw_data[(buffer_location + 5):(buffer_location + 8 * num_events + 5):8].astype(np.uint16) << 8) + (raw_data[(buffer_location + 4):(buffer_location + 8 * num_events + 4):8]))
        t = ((raw_data[(buffer_location + 7):(buffer_location + 8 * num_events + 7):8].astype(np.uint32) << 8) + (raw_data[(buffer_location + 6):(buffer_location + 8 * num_events + 6):8]))
        buffer_location += num_events * 8
        t += start_time
        overflows = np.where(event_type == 2)

        for i in range(len(overflows[0])):
            overflow_loc = overflows[0][i]
            t[overflow_loc:] = t[overflow_loc:] + 65536

        locations = np.where((event_type == 0) | (event_type == 3))
        indices[start_index:(start_index + num_events)][locations] = True
        full_x[start_index:(start_index + num_events)] = x
        full_y[start_index:(start_index + num_events)] = y
        full_t[start_index:(start_index + num_events)] = t
        full_p[start_index:(start_index + num_events)] = event_subtype
        full_f[start_index:(start_index + num_events)] = event_type

        start_index += num_events

    processed_events.events["x"] = full_x[indices]
    processed_events.events["y"] = full_y[indices]
    processed_events.events["t"] = full_t[indices]
    processed_events.events["p"] = full_p[indices]

    # Polarity need to be flipped i.e. 0s become 1s and 1s become 0
    processed_events.events["p"] = np.abs(processed_events.events["p"] - 1)

    return processed_events
