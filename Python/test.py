from tkinter import N
from src.harris_detector import *
from math import pi, exp

WIDTH = 240
HEIGHT = 180

def factorial(n):
    if n > 1:
        return factorial(n-1)*n
    else:
        return 1

def pascal(k, n):
    if 0 <= k <= n:
        return factorial(n)/(factorial(n-k)*factorial(k))
    else:
        return 0

def main():
    #test_image = np.array([[-1, 0, 1, -1, 0, 1, -1, 0, 1],
    #                    [-2, 0, 2, -1, 0, 1, -1, 0, 1],
    #                    [-1, 0, 1, -1, 0, 1, -1, 0, 1],
    #                    [-1, 0, 1, -1, 0, 1, -1, 0, 1],
    #                    [-1, 0, 1, -1, 0, 1, -1, 0, 1],
    #                    [-1, 0, 1, -1, 0, 1, -1, 0, 1],
    #                    [-1, 0, 1, -1, 0, 1, -1, 0, 1],
    #                    [-1, 0, 1, -1, 0, 1, -1, 0, 1],
    #                    [-1, 0, 1, -1, 0, 1, -1, 0, 1]])
    kernel_size = 5
    window_size = 4
    smooth_function = np.zeros(kernel_size, dtype=np.int32)
    difference_function = np.zeros(kernel_size, dtype=np.int32)

    for i in range(kernel_size):
        smooth_function[i] = factorial(kernel_size - 1)/(factorial(kernel_size-1-i) * factorial(i))
        difference_function[i] = pascal(i, kernel_size - 2) - pascal(i-1, kernel_size - 2)

    print(smooth_function)
    print(difference_function)

    #Gx = smooth_function * np.transpose(difference_function)
    Gx = smooth_function * difference_function
    Gx /= max(Gx)

    #Gaussian_window
    sigma = 1
    coeff = 1/(2*pi*sigma**2)
    l = (2*window_size+2-kernel_size)//2
    window = np.zeros((2*l+1, 2*l+1), dtype=np.uint32)
    for x in range(-l, l+1):
        for y in range(-l, l+1):
            window[y][x] = coeff*exp(-(x**2 + y**2)/(2 * sigma**2))
    window //= sum(window)

    

if __name__ == "__main__":
    main()