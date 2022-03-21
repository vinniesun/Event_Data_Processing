import cv2
import numpy as np

IMAGE_PATH_1B1C = "../images/1b1c/20180712_Site2_3pm_6mm_01/16"
IMAGE_PATH_1B2C = "../images/1b2c/20180711_Site1_3pm_12mm_01/33.jpg"
HEIGHT = 180
WIDTH = 240
CONNECTIVITY = 4

def medianFilter(rawIm):
    filtered = cv2.medianBlur(rawIm, 3) # Ksize aperture linear must be odd and greater than 1 ie 3, 5, 7... 
    
    return filtered

"""
    @param
    - src: a greyscale image (cannot be RGB, OpenCV does not handle RGB image with connectedComponentsWithStats())

    @return
    - numLabels: Number of unique labels
    - labels: a mask that has the same spatial dimensions as our input image.
              For each location in labels, we have an integer ID value that corresponds to the connected component where the pixel belongs.
    - stats: statistics on each connected component, including the bounding box coordinates and area in pixels
    - centroid: The center (x, y) coordinates of each connected components
"""
def CCL(src):
    numLabels, labels, stats, centroids = cv2.connectedComponentsWithStats(src, CONNECTIVITY, cv2.CV_32S) # 8-ways or 4-ways connectivity

    return numLabels, labels, stats, centroids

def drawCCL(src, numLabels, labels, stats, centroids):
    result = src.copy()
    #colours = []
    componentMask = []

    #for i in range(0, numLabels):
    #    if i == 0:
    #        colours.append((0, 0, 0))
    #    else:
    #        colours.append((int(255/numLabels*i), int(255/numLabels*i), int(255/numLabels*i)))
    
    # the first label ID, 0, is always the background, which we will ignore
    for i in range(0, numLabels):
        if i == 0:
            continue
        else:
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            area = stats[i, cv2.CC_STAT_AREA]
            cx, cy = centroids[i]
            cv2.rectangle(result, (x, y), (x+w, y+h), (0, 255, 0), 3)
            #cv2.rectangle(result, (x, y), (x+w, y+h), colours[i], 3)
            #cv2.circle(result, (int(cx), int(cy)), 4, colours[i], -1)
            componentMask.append((labels==i).astype("uint8") * 255)
    
    return result, componentMask

def process1b1c():
    src = cv2.imread(IMAGE_PATH_1B1C+".jpg")
    src = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    filtered = cv2.medianBlur(src, 3)
    #output = cv2.cvtColor(filtered, cv2.COLOR_GRAY2BGR)

    numLabels, labels, stats, centroids = CCL(filtered)
    result, componentMasks = drawCCL(filtered, numLabels, labels, stats, centroids)

    cv2.imwrite(IMAGE_PATH_1B1C + "_filtered.jpg", result)
    cv2.imshow("Original Image", src)
    cv2.imshow("Filtered Image", filtered)
    cv2.imshow("labels", result)
    cv2.waitKey(0)

    cv2.destroyAllWindows()

def process1b2c():
    src = cv2.imread(IMAGE_PATH_1B2C)
    #src = cv2.imread(IMAGE_PATH_1B2C, cv2.IMREAD_GRAYSCALE)

    filtered = cv2.medianBlur(src, 3)
    cv2.imwrite("../images/1b2c/20180711_Site1_3pm_12mm_01/33_filtered.jpg", filtered)

    cv2.imshow("Original Image", src)
    cv2.imshow("Filtered Image", filtered)
    cv2.waitKey(0)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    process1b1c()
    #process1b2c()
