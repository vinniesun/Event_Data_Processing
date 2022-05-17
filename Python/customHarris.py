from concurrent.futures import process
import numpy as np
from scipy.signal import convolve2d
import cv2
from matplotlib import pyplot as plt
from src.process_events import *

# np.float64 == np.float_, which is the same as C's double
NUMPY_DTYPE_ARRAY = [np.uint8, np.uint16, np.uint32, np.uint64, np.float16, np.float32, np.float64]
INPUT_FILE = "/Users/vincent/Desktop/CityUHK/Event_Process/Event_Camera_Dataset/shapes_rotation/events.txt"
# INPUT_FILE = "/Users/vincent/Desktop/CityUHK/Event_Process/Event_Camera_Dataset/shapes_translation/events.txt"
# INPUT_FILE = "/Users/vincent/Desktop/CityUHK/Event_Process/Event_Camera_Dataset/shapes_6dof/events.txt"
# INPUT_FILE = "/Users/vincent/Desktop/CityUHK/Event_Process/Event_Camera_Dataset/poster_rotation/events.txt"

def harrisCorner1(input, dtype) -> np.array:
    # np.savetxt("python_Input.csv", input, delimiter=",")
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=dtype)
    sobel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=dtype)

    gauss = np.array([[1,2,1],[2,4,2],[1,2,1]], dtype=dtype)
    factor = 1/16
    gauss *= factor

    Ix = convolve2d(input, sobel_x, "same")
    Iy = convolve2d(input, sobel_y, "same")
        # print(Ix.dtype)
    # print(Iy.dtype)
    # np.savetxt("python_Ix.csv", Ix, delimiter=",")

    Ixy = Ix * Iy
    Ixx = Ix ** 2
    Iyy = Iy ** 2
    # print(Ixy.dtype)
    # print(Ixx.dtype)
    # print(Iyy.dtype)
    # np.savetxt("python_Ixx.csv", Ixx, delimiter=",")

    gIxx = convolve2d(Ixx, gauss, "same")
    gIyy = convolve2d(Iyy, gauss, "same")
    gIxy = convolve2d(Ixy, gauss, "same")
    # print(Ixy.dtype)
    # print(Ixx.dtype)
    # print(Iyy.dtype)
    # np.savetxt("python_gIxx.csv", gIxx, delimiter=",")

    det = gIxx*gIyy - gIxy**2
    trace = ((gIxx+gIyy)**2)*0.04
    response = det - trace
    # print(det.dtype)
    # print(trace.dtype)
    # print(response.dtype)
    # np.savetxt("python_Harris_Response.csv", response, delimiter=",")

    return response

def harrisCorner2(input: np.array, blockSize: int, dtype) -> np.array:
    height, width = input.shape
    offset = int(blockSize/2)
    # np.savetxt("python_Input.csv", input, delimiter=",")
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=dtype)
    sobel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=dtype)

    gauss = np.array([[1,2,1],[2,4,2],[1,2,1]], dtype=dtype)
    factor = 1/16
    gauss *= factor

    Ix = convolve2d(input, sobel_x, "same")
    Iy = convolve2d(input, sobel_y, "same")
        # print(Ix.dtype)
    # print(Iy.dtype)
    # np.savetxt("python_Ix.csv", Ix, delimiter=",")

    Ixy = Ix * Iy
    Ixx = Ix ** 2
    Iyy = Iy ** 2
    # print(Ixy.dtype)
    # print(Ixx.dtype)
    # print(Iyy.dtype)
    # np.savetxt("python_Ixx.csv", Ixx, delimiter=",")

    gIxx = convolve2d(Ixx, gauss, "same")
    gIyy = convolve2d(Iyy, gauss, "same")
    gIxy = convolve2d(Ixy, gauss, "same")
    # print(Ixy.dtype)
    # print(Ixx.dtype)
    # print(Iyy.dtype)
    # np.savetxt("python_gIxx.csv", gIxx, delimiter=",")

    response = np.zeros((height, width), dtype=dtype)

    for y in range(offset, height-offset):
        for x in range(offset, width-offset):
            Sxx = np.sum(gIxx[y-offset:y+offset+1, x-offset:x+offset+1])
            Syy = np.sum(gIyy[y-offset:y+offset+1, x-offset:x+offset+1])
            Sxy = np.sum(gIxy[y-offset:y+offset+1, x-offset:x+offset+1])

            det = (Sxx*Syy) - (Sxy**2)
            trace = ((Sxx + Syy)**2)*0.04

            response[y][x] = det - trace

    # print(det.dtype)
    # print(trace.dtype)
    # print(response.dtype)
    # np.savetxt("python_Harris_Response.csv", response, delimiter=",")

    return response

def test():
    input = np.array([[0,0,0,25,25,27,27,27,27],
                      [0,0,0,25,25,27,27,27,27],
                      [0,0,0,25,25,27,27,27,27],
                      [25,25,25,25,25,27,27,27,27],
                      [25,25,25,25,25,27,27,27,27],
                      [25,25,25,25,25,27,27,27,27],
                      [25,25,25,25,25,27,27,27,27],])
    
    input_img = cv2.imread("1kjl.jpg", cv2.IMREAD_GRAYSCALE)
    colour_img = cv2.imread("1kjl.jpg")
    output = harrisCorner2(input_img, 2, NUMPY_DTYPE_ARRAY[-1])

    maxVal = np.amax(output)
    print(maxVal)

    for row in range(len(input_img)):
        for col in range(len(input_img[0])):
            if output[row][col] > maxVal*0.6:
                colour_img = cv2.circle(colour_img, (col, row), 3, (0, 0, 255), 2) # Colour order is B,G,R

    cv2.imwrite("custom_python_output.jpg", colour_img)

    histogramAnalysis(output, "aklsjf")

def histogramAnalysis(harrisResponse: np.array, name: str) -> None:
    hist, bin_edges = np.histogram(harrisResponse, bins=10)
    #plt.bar(bin_edges[:-1], hist, width=0.5, color='blue')
    n, bins, patches = plt.hist(x=harrisResponse, bins=30, alpha=0.7, rwidth=0.85)
    plt.title('histogram')
    plt.savefig("./Output/histogram_"+name+".jpg")
    plt.close()

def main(filename: str, dtype):
    current_events = process_text_file(filename)

    current_events.events, current_events.num_events = refractory_filtering(current_events, refractory_period=1000)
    print("After refractory filtering, number of events remaining is: ", current_events.num_events)
    print("---------------------------------------------------------------")

    current_events.events, current_events.num_events = background_activity_filter(current_events, time_window=5000)
    print("After NN filtering, number of events remaining is: ", current_events.num_events)
    print("---------------------------------------------------------------")

    tos = np.zeros((current_events.height, current_events.width), dtype=np.uint8)

    ktos = 3
    ttos = 2*(2*ktos + 1)

    response = np.zeros((current_events.height, current_events.width), dtype=dtype)
    output_img = np.zeros((current_events.height, current_events.width, 3), dtype=np.uint8)

    prev_time = 0

    for i in tqdm(range(400000, 500000)):
        x = current_events.events[i]["x"]
        y = current_events.events[i]["y"]
        t = current_events.events[i]["t"]
        p = current_events.events[i]["p"]

        tos = update_time_surface(tos, t, x, y, p, ktos=ktos, ttos=ttos, maxX=current_events.width, maxY=current_events.height, mode="TOS")

        if i % 20 == 0:
            response = harrisCorner1(tos, dtype)
        
        maxVal = np.amax(response)
        if response[y][x] > maxVal * 0.8:
            output_img = cv2.cvtColor(tos, cv2.COLOR_GRAY2BGR)
            cv2.circle(output_img, (x, y), 3, (255, 0, 0), 1)

            cv2.imwrite("./Output/"+str(i)+".jpg", output_img)

            histogramAnalysis(response, str(i))

        #if i%50 == 0:
        #    cv2.imwrite("./Output/"+str(i)+"_grayscale.jpg", tos)

if __name__ == "__main__":
    main(INPUT_FILE, NUMPY_DTYPE_ARRAY[-1])
    #test()