from collections import deque
import numpy as np
from math import pi, exp
from util import crop

# The HEIGHT and WIDTH of a DAVIS Camera
WIDTH = 240
HEIGHT = 180

class eHarris():
    def __init__(self, pixelSize=9, queueSize=25, harrisThreshold=8.0, windowSize=4, kernelSize=5):
        self.queue = deque(maxlen=25)
        self.harrisThreshold = harrisThreshold
        self.windowSize = windowSize
        self.kernelSize = kernelSize
        self.last_score = None
        self.Sx = np.empty(kernelSize)
        self.Dx = np.empty(kernelSize)

        for i in range(kernelSize):
            self.Sx[i] = self.factorial(kernelSize-1)/(self.factorial(kernelSize-1-i)*self.factorial(i))
            self.Dx[i] = self.pasc(i, kernelSize-2) - self.pasc(i-1, kernelSize-2)
        
        self.Gx = self.Sx * np.transpose(self.Dx)
        self.Gx = self.Gx / np.max(self.Gx)

        self.sigma = 1
        self.A = 1.0/(2.0*pi*self.sigma**2)
        self.l2 = (2*windowSize + 2 - kernelSize)/2
        self.h = np.array((2*self.l2+1, 2*self.l2+1))

        for row in range(-1*self.l2, self.l2+1):
            for col in range(-1*self.l2, self.l2+1):
                h_xy = self.A * exp(-(row**2 + col**2)/(2*self.sigma**2))
                self.h[self.l2+row][self.l2+col] = h_xy

        self.h /= np.sum(self.h)

    def getLastScore(self):
        return self.last_score

    def factorial(self, n):
        if n>1:
            return n*self.factorial(n-1)
        else:
            return 1

    def pasc(self, k, n):
        if k>=0 and k<=n:
            return self.factorial(n)/(self.factorial(n-k)*self.factorial(k))
        else:
            return 0

    def getHarrisScore(self, image, centerX, centerY, pol):
        # Check if it's at the border
        if (centerX < self.windowSize or centerX > WIDTH - self.windowSize or centerY < self.windowSize or centerY > HEIGHT - self.windowSize):
            return self.harrisThreshold - 10.0

        local_frame = crop(image, centerX, centerY, self.windowSize)

        l = 2*self.windowSize + 2 - self.kernelSize
        dx = np.zeros((l, l))
        dy = np.zeros((l, l))

        for row in range(l):
            for col in range(l):
                for krow in range(self.kernelSize):
                    for kcol in range(self.kernelSize):
                        dx[row][col] += local_frame[row+krow][col+kcol] * self.Gx[krow][kcol]
                        dx[row][col] += local_frame[row+krow][col+kcol] * self.Gx[kcol][krow]

        a, b, d = 0.0, 0.0, 0.0
        for row in range(l):
            for col in range(l):
                a += self.h[row][col] * dx[row][col] * dx[row][col]
                b += self.h[row][col] * dx[row][col] * dy[row][col]
                d += self.h[row][col] * dy[row][col] * dy[row][col]

        score = a*d - b**2 - 0.04*(a+d)**2

        return score

    def isFeature(self, image, centerX, centerY, pol):
        score = self.harrisThreshold - 10.0

        if self.queue[-1]: # the queue is full
            self.queue.popleft()
            self.queue.append((centerX, centerY, pol))
            score = self.getHarrisScore()
            self.last_score = score
        else:
            self.queue.append((centerX, centerY, pol))
            if self.queue[-1]:
                score = self.getHarrisScore(image[:,:,pol], centerX, centerY, pol)
                self.last_score = score

        return score > self.harrisThreshold