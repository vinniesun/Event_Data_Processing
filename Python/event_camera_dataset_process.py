import csv
import cv2

from src.process_events import *
from src.plot_tools import *
from src.arcstar import *
from src.efast import *
from src.util import *

WIDTH = 240
HEIGHT = 180
ACCUMULATED_TIME = 33000
REFRACTORY_PERIOD = 1000 # Original is 1000
NN_WINDOW = 5000 # Original is 500
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
            events.events[i]["t"], events.events[i]["x"], events.events[i]["y"], events.events[i]["p"] = int(float(event[0]) * 1e6), int(event[1]), int(event[2]), bool(event[3])
            
    return events

def main():
    #filename = "/Users/vincent/Desktop/CityUHK/Event_Process/Event_Camera_Dataset/shapes_rotation/events.txt"
    filename = "/Users/vincent/Desktop/CityUHK/Event_Process/Event_Camera_Dataset/shapes_translation/events.txt"
    #filename = "/Users/vincent/Desktop/CityUHK/Event_Process/Event_Camera_Dataset/shapes_6dof/events.txt"
    #filename = "/Users/vincent/Desktop/CityUHK/Event_Process/Event_Camera_Dataset/poster_rotation/events.txt"

    """
    eFast_csv_file = "../Output/eFast_Result.csv"
    ArcStar_csv_file = "../Output/ArcStar_Result.csv"

    with open(eFast_csv_file, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["x", "y", "t", "p"])
    print("eFast Result File Created")

    with open(ArcStar_csv_file, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["x", "y", "t", "p"])
    print("ArcStar Result File Created")
    print("---------------------------------------------------------------")
    """
    # Read the event text file and generate our stream of events
    current_events = process_text_file(filename)
    print("Total Number of Events extracted: " + str(current_events.num_events))
    print("---------------------------------------------------------------")
    
    # Apply refractory filtering
    current_events.events, current_events.num_events = refractory_filtering(current_events, refractory_period=REFRACTORY_PERIOD)
    print("Total Number of Events After Refractory Filtering: " + str(current_events.num_events))
    print("---------------------------------------------------------------")

    # Apply nearest neighbourhood/background activity filtering
    current_events.events, current_events.num_events = background_activity_filter(current_events, time_window=NN_WINDOW)
    print("Total Number of Events After Nearest Neighbourhood Filtering: " + str(current_events.num_events))
    print("---------------------------------------------------------------")    

    """
    # Find all Corners & All Events Feature Track
    # Generate Time Surface
    start_event_number = 200000
    end_event_number = 300000
    eFastQueue, ArcStarQueue = [], []
    features = []
    on_count, off_count = 0, 0
    #time_surface = generate_time_surface(current_events, ACCUMULATED_TIME, event_end=start_event_number, mode="delta")
    time_surface = np.zeros((current_events.height, current_events.width, 2), dtype=np.int64)
    eFast_sub, arcstar_sub = [], []
    prev_time = current_events.events[start_event_number-1]["t"]
    #prev_time = 0

    for i, e in tqdm(enumerate(current_events.events[start_event_number:end_event_number])):
        #time_surface = update_time_surface(time_surface, e["t"], e["x"], e["y"], e["p"], mode="absolute")
        #time_surface = update_time_surface(time_surface, e["t"], e["x"], e["y"], e["p"], time_threshold=ACCUMULATED_TIME, prev_time=prev_time, mode="delta")
        #time_surface = update_time_surface(time_surface, e["t"], e["x"], e["y"], e["p"], time_threshold=ACCUMULATED_TIME, prev_time=prev_time, bits=24, mode="bits")
        
        #time_surface[e["y"]][e["x"]][int(e["p"])] = e["t"]

        deltaT = e["t"] - prev_time
        time_surface[:, :, int(e["p"])] -= deltaT
        time_surface[e["y"]][e["x"]][int(e["p"])] = ACCUMULATED_TIME
        mask = np.where(time_surface <= 0)
        time_surface[mask] = 0

        prev_time = e["t"]
        prev_state, prev_state_inv = time_surface[e["y"]][e["x"]][int(e["p"])], time_surface[e["y"]][e["x"]][int(not e["p"])]
        isEFast = isCornerEFast(time_surface[:, :, int(e["p"])], e["x"], e["y"], e["p"])
        isArcStar = isCornerArcStar(time_surface[:, :, int(e["p"])], prev_state, prev_state_inv, e["x"], e["y"], e["p"])

        if isEFast:
            eFastQueue.append((e["x"], e["y"], e["t"], int(e["p"])*255))
            if len(eFast_sub) < 60:
                eFast_sub.append((time_surface[:, :, int(e["p"])].copy(), e["x"], e["y"], abs(i - end_event_number), int(e["p"])))
            with open(eFast_csv_file, 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([e["x"], e["y"], e["t"], e["p"]])

        if isArcStar:
            ArcStarQueue.append((e["x"], e["y"], e["t"], int(e["p"])*255))
            if len(arcstar_sub) < 60:
                arcstar_sub.append((time_surface[:, :, int(e["p"])].copy(), e["x"], e["y"], abs(i - end_event_number), int(e["p"])))
            with open(ArcStar_csv_file, 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([e["x"], e["y"], e["t"], e["p"]])

        if e["p"]:
            on_count += 1
        else:
            off_count += 1
        
        features.append((e["x"], e["y"], e["t"], int(e["p"])*255))
    
    # Draw All Events Feature Track
    drawFeatureTrack3D(features, "Shapes Translation All Events Feature Track", ACCUMULATED_TIME)
    drawFeatureTrack2D(features, "Shapes Translation All Events Feature Track", ACCUMULATED_TIME)
    print("Number of On event in this feature track: " + str(on_count))
    print("Number of Off event in this feature track: " + str(off_count))
    print("---------------------------------------------------------------")

    # Draw All Corner's Feature Track
    print("Number of eFast Corners in this feature track: " + str(len(eFastQueue)))
    print("Number of ArcStar Corners in this feature track: " + str(len(ArcStarQueue)))
    drawFeatureTrack3D(eFastQueue, "Shapes Translation All eFast Corner in File", ACCUMULATED_TIME)
    drawFeatureTrack2D(eFastQueue, "Shapes Translation All eFast Corner in File", ACCUMULATED_TIME)
    drawFeatureTrack3D(ArcStarQueue, "Shapes Translation All ArcStar Corner in File", ACCUMULATED_TIME)
    drawFeatureTrack2D(ArcStarQueue, "Shapes Translation All ArcStar Corner in File", ACCUMULATED_TIME)
    print("---------------------------------------------------------------")

    # Plot All eFast Response
    noRow, noCol = 5, 4
    fig = plt.figure(figsize=(30, 30))
    plt.tight_layout()
    for j, (img, locX, locY, recEventNo, p) in enumerate(eFast_sub[-20:]):
        ax = fig.add_subplot(noRow, noCol, j+1) 
        temp = crop(img, locX, locY, 9)
        ax = drawHeatMapSub(temp, recEventNo, subplot=ax, title=str(locY) + ", " + str(locX))
    plt.savefig("../Output/" + str(5000) + "us 2D eFast Only, Corners detected - " + str(len(eFastQueue)) + ".jpg", dpi=300)
    plt.show()
    plt.close(fig)

    # Plot All eFast Response as Binarised Images
    noRow, noCol = 5, 4
    fig = plt.figure(figsize=(30, 30))
    plt.tight_layout()
    for j, (img, locX, locY, recEventNo, p) in enumerate(eFast_sub[-20:]):
        ax = fig.add_subplot(noRow, noCol, j+1) 
        temp = crop(img, locX, locY, 9)
        threshold = np.max(temp)
        mask = np.where(temp > threshold*0.90)
        temp[:, :] = 0
        temp[mask] = 1
        ax = drawHeatMapSub(temp, recEventNo, subplot=ax, title=str(locY) + ", " + str(locX))
    plt.savefig("../Output/" + str(5000) + "us 2D binary eFast Only, Corners detected - " + str(len(eFastQueue)) + ".jpg", dpi=300)
    plt.show()
    plt.close(fig)

    # Plot All ArcStar Response
    noRow, noCol = 5, 4
    fig = plt.figure(figsize=(30, 30))
    plt.tight_layout()
    for j, (img, locX, locY, recEventNo, p) in enumerate(arcstar_sub[-20:]):
        ax = fig.add_subplot(noRow, noCol, j+1)
        temp = crop(img, locX, locY, 9)
        ax = drawHeatMapSub(temp, recEventNo, subplot=ax, title=str(locY) + ", " + str(locX))
    plt.savefig("../Output/" + str(5000) + "us 2D ArcStar Only, Corners detected - " + str(len(ArcStarQueue)) + ".jpg", dpi=300)
    plt.show()
    plt.close(fig)

    # Plot All ArcStar Response as Binarised Images
    noRow, noCol = 5, 4
    fig = plt.figure(figsize=(30, 30))
    plt.tight_layout()
    for j, (img, locX, locY, recEventNo, p) in enumerate(arcstar_sub[-20:]):
        ax = fig.add_subplot(noRow, noCol, j+1)
        temp = crop(img, locX, locY, 9)
        threshold = np.max(temp)
        mask = np.where(temp > threshold*0.90)
        temp[:, :] = 0
        temp[mask] = 1
        ax = drawHeatMapSub(temp, recEventNo, subplot=ax, title=str(locY) + ", " + str(locX))
    plt.savefig("../Output/" + str(5000) + "us 2D binary ArcStar Only, Corners detected - " + str(len(ArcStarQueue)) + ".jpg", dpi=300)
    plt.show()
    plt.close(fig)
    """
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