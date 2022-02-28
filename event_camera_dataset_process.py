from turtle import back
import cv2

from src.process_events import *
from src.plot_tools import *

WIDTH = 240
HEIGHT = 180
output = "../Output/"

def addChannel(image):
    no_of_frames = image.shape[0]
    dummy_channel = np.zeros((no_of_frames, HEIGHT, WIDTH, 1), dtype=np.uint8)
    print(dummy_channel.shape)
    image = np.append(image, dummy_channel, axis=3)
    print(image.shape)
	
    return image

def display_video(images) -> None:
    flag = True
    while flag:
        for img in images:
            cv2.imshow('frame', img)

            if cv2.waitKey(60) & 0xFF == ord('q'):
                flag = False
                break

    cv2.destroyAllWindows()

def save_video(frames, mode: str) -> None:
    # Store the output as a video
    img_array = []
    for f in frames:
        #print(f.shape, type(f))
        height, width, _ = f.shape
        size = (width, height)
        img_array.append(f)
    
    vid_recorder = cv2.VideoWriter(output+mode+' shapes.avi', cv2.VideoWriter_fourcc(*'DIVX'), 15, size)

    for i in img_array:
        vid_recorder.write(i)
    vid_recorder.release()

def process_text_file(filename: str) -> Events:
    with open(filename, 'r') as f:
        num_events = 0
        for _ in f:
            num_events += 1
    
    events = Events(num_events, WIDTH, HEIGHT)

    with open(filename, 'r') as f:
        for i, line in enumerate(tqdm(f)):
            event = line.split(" ")
            assert len(event) == 4, "the line should contain only four elements: t, x, y, p"
            events.events[i]["t"], events.events[i]["x"], events.events[i]["y"], events.events[i]["p"] = int(float(event[0]) * 10e6), int(event[1]), int(event[2]), bool(event[3])
            
    return events

def main():
    #filename = "/Users/vincent/Desktop/CityUHK/Event_Camera_Dataset/shapes_rotation/events.txt"
    filename = "/Users/vincent/Desktop/CityUHK/Event_Camera_Dataset/shapes_translation/events.txt"

    # Read the event text file and generate our stream of events
    current_events = process_text_file(filename)
    print("Total Number of Events extracted: " + str(current_events.num_events))
    print("---------------------------------------------------------------")

    # Generate Time Surface
    #time_surface = generate_time_surface(current_events, 5000, event_start=0, event_end=5000)
    #print(time_surface.shape)
    #print("---------------------------------------------------------------")
    
    # Apply refractory filtering
    #current_events.events, current_events.num_events = refractory_filtering(current_events, refractory_period=500)
    #print("Total Number of Events After Refractory Filtering: " + str(current_events.num_events))
    #print("---------------------------------------------------------------")

    # Apply nearest neighbourhood/background activity filtering
    #current_events.events, current_events.num_events = background_activity_filter(current_events, time_window=500)
    #print("Total Number of Events After Nearest Neighbourhood Filtering: " + str(current_events.num_events))
    #print("---------------------------------------------------------------")

    # Feature Tracks
    features = []
    on_count, off_count = 0, 0
    for i in tqdm(range(50000)):
        if current_events.events[i]["p"]:
            on_count += 1
        else:
            off_count += 1
        features.append((current_events.events[i]["x"], current_events.events[i]["y"], int(current_events.events[i]["p"])*255))
    drawFeatureTrack3D(features, "", 33000)
    drawFeatureTrack2D(features, "", 33000)
    print("Number of On event in this feature track: " + str(on_count))
    print("Number of Off event in this feature track: " + str(off_count))
    print("---------------------------------------------------------------")

    # generate frames based on accumulated time
    sliced = time_window_slice(current_events, time_window=33000)
    frames = accumulate_and_generate(sliced, WIDTH, HEIGHT)
    print("---------------------------------------------------------------")

    # add a dummy channel to the frame so it can be displayed with OpenCV
    frames = addChannel(frames)
    print("---------------------------------------------------------------")

    # Display the resulting frames as a video
    #display_video(frames)
    save_video(frames, "filtered")

if __name__ == "__main__":
    main()