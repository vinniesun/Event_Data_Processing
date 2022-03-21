import numpy as np

# Size can be either 5 or 10
def crop(image, startX, startY, size):
    half = size // 2
    return image[startY-half:startY+(size-half), startX-half:startX+(size-half)]

def getMaxIndex(image):
    maxIndices = np.where(image == np.amax(image))
    result = list(zip(maxIndices[0], maxIndices[1]))

    return result

def medianFilter(data, filter_size):
    #filtered = cv2.medianBlur(rawIm, 3) # Ksize aperture linear must be odd and greater than 1 ie 3, 5, 7... 
    
    #return filtered
    temp = []
    indexer = filter_size // 2
    data_final = []
    data_final = np.zeros((len(data),len(data[0])))
    for i in range(len(data)):
        for j in range(len(data[0])):
            for z in range(filter_size):
                if i + z - indexer < 0 or i + z - indexer > len(data) - 1:
                    for c in range(filter_size):
                        temp.append(0)
                else:
                    if j + z - indexer < 0 or j + indexer > len(data[0]) - 1:
                        temp.append(0)
                    else:
                        for k in range(filter_size):
                            temp.append(data[i + z - indexer][j + k - indexer])

            temp.sort()
            data_final[i][j] = temp[len(temp) // 2]
            temp = []
    return data_final

def normalise(image):
    oldMin = 0
    oldMax = image.max()
    newMax = 255

    normalised = (image - oldMin)/(oldMax - oldMin)
    normalised *= newMax
    normalised = normalised.astype(np.uint8)

    return normalised