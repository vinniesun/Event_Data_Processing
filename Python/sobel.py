import numpy as np

def factorial(n: int):
    if n > 1:
        return int(n*factorial(n-1))
    else:
        return 1

def pascal(k: int, n: int):
    if (k >= 0 and k <= n):
        return int(factorial(n)/(factorial(n-k)*factorial(k)))
    else:
        return 0

if __name__ == '__main__':
    kernel_size = 5
    Sx, Dx = np.zeros(kernel_size, dtype=np.int32), np.zeros(kernel_size, dtype=np.int32)

    for i in range(kernel_size):
        print(i)
        Sx[i] = factorial(kernel_size - 1) / (factorial(kernel_size - 1 - i) * factorial(i))
        print(Sx[i])
        Dx[i] = pascal(i, kernel_size - 2) - pascal(i-1, kernel_size - 2)
        print(Dx[i])
        print("-------------------------------------")

    sobel = np.zeros((kernel_size, kernel_size), dtype=np.int32)
    for r in range(kernel_size):
        for c in range(kernel_size):
            sobel[r][c] = Sx[r]*Dx[c]

    print(sobel)

    sobel = np.zeros((kernel_size, kernel_size), dtype=np.int32)
    for r in range(kernel_size):
        for c in range(kernel_size):
            sobel[r][c] = Sx[c]*Dx[r]

    print(sobel)

    Gx = Sx * Dx.transpose()
    Gx = Gx / Gx.max()

    print(Sx)
    print(Dx)
    print(Gx)