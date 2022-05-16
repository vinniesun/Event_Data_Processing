import numpy as np
from scipy.signal import convolve2d
import cv2
from matplotlib import pyplot as plt

# np.float64 == np.float_, which is the same as C's double
NUMPY_DTYPE_ARRAY = [np.uint8, np.uint16, np.uint32, np.uint64, np.float16, np.float32, np.float64]

def process(input, dtype) -> np.array:
    np.savetxt("python_Input.csv", input, delimiter=",")
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=dtype)
    sobel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=dtype)

    gauss = np.array([[1,2,1],[2,4,2],[1,2,1]], dtype=dtype)
    factor = 1/16
    gauss *= factor

    Ix = convolve2d(input, sobel_x, "same")
    print(Ix.dtype)
    Iy = convolve2d(input, sobel_y, "same")
    print(Iy.dtype)
    #np.savetxt("python_Ix.csv", Ix, delimiter=",")

    Ixy = Ix * Iy
    Ixx = Ix ** 2
    Iyy = Iy ** 2
    print(Ixy.dtype)
    print(Ixx.dtype)
    print(Iyy.dtype)
    #np.savetxt("python_Ixx.csv", Ixx, delimiter=",")

    gIxx = convolve2d(Ixx, gauss, "same")
    gIyy = convolve2d(Iyy, gauss, "same")
    gIxy = convolve2d(Ixy, gauss, "same")
    print(Ixy.dtype)
    print(Ixx.dtype)
    print(Iyy.dtype)
    #np.savetxt("python_gIxx.csv", gIxx, delimiter=",")

    det = gIxx*gIyy - gIxy**2
    trace = ((gIxx+gIyy)**2)*0.04
    response = det - trace
    print(det.dtype)
    print(trace.dtype)
    print(response.dtype)
    #np.savetxt("python_Harris_Response.csv", response, delimiter=",")

    return response

if __name__ == "__main__":
    input = np.array([[0,0,0,25,25,27,27,27,27],
                      [0,0,0,25,25,27,27,27,27],
                      [0,0,0,25,25,27,27,27,27],
                      [25,25,25,25,25,27,27,27,27],
                      [25,25,25,25,25,27,27,27,27],
                      [25,25,25,25,25,27,27,27,27],
                      [25,25,25,25,25,27,27,27,27],])
    
    input_img = cv2.imread("1kjl.jpg", cv2.IMREAD_GRAYSCALE)
    colour_img = cv2.imread("1kjl.jpg")
    output = process(input_img, NUMPY_DTYPE_ARRAY[-1])

    maxVal = np.amax(output)
    print(maxVal)

    for row in range(len(input_img)):
        for col in range(len(input_img[0])):
            if output[row][col] > maxVal*0.6:
                colour_img = cv2.circle(colour_img, (col, row), 3, (0, 0, 255), 2) # Colour order is B,G,R

    cv2.imwrite("custom_python_output.jpg", colour_img)

    hist, bin_edges = np.histogram(output)
    print(hist)
    print(bin_edges)

    #plt.xlim(min(bin_edges), max(bin_edges))
    #plt.bar(bin_edges[:-1], hist, width=0.5, color='blue')
    n, bins, patches = plt.hist(x=output, bins='auto', alpha=0.7, rwidth=0.85)
    plt.title('histogram')
    plt.show()

    print(n)
    print(bins)
    print(patches)

    #r = process(input)
    #print(r)