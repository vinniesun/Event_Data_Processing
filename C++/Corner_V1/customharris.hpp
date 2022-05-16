#ifndef CUSTOMHARRIS_H
#define CUSTOMHARRIS_H

#include <iostream>
#include <vector>
#include <cmath>
#include <chrono> // For timing the execution time
#include <fstream>
#include "opencv2/opencv.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/core.hpp"

template <class T>
class HarrisCornerDetector {
public:
    HarrisCornerDetector(int kSize, int wSize, int height, int width);
    ~HarrisCornerDetector();

    int pascal(int k, int n);
    int factorial(int n);
    void square(std::vector<std::vector<T>> &input);
    void multiply(const std::vector<std::vector<T>> &input1, const std::vector<std::vector<T>> &input2, std::vector<std::vector<T>> &output);
    void multFactor(std::vector<std::vector<T>> &input);
    void subtract(const std::vector<std::vector<T>> &input1, const std::vector<std::vector<T>> &input2, std::vector<std::vector<T>> &output);
    void add(const std::vector<std::vector<T>> &input1, const std::vector<std::vector<T>> &input2, std::vector<std::vector<T>> &output);
    void Conv2D(const std::vector<std::vector<int>> &input, std::vector<std::vector<T>> &output, const std::vector<std::vector<T>> &kernel, std::string mode);
    void Conv2D(const std::vector<std::vector<T>> &input, std::vector<std::vector<T>> &output, const std::vector<std::vector<T>> &kernel, std::string mode);
    void separableConv2D(const std::vector<std::vector<int>> &input, std::vector<std::vector<T>> &output, const std::vector<T> &kernel_h, const std::vector<T> &kernel_v, std::string mode);
    void separableConv2D(const std::vector<std::vector<T>> &input, std::vector<std::vector<T>> &output, const std::vector<T> &kernel_h, const std::vector<T> &kernel_v, std::string mode);
    //static std::vector<std::vector<int>> transpose(std::vector<std::vector<int>> &input);
    void generateSobel();
    void generateHarrisScore(const std::vector<std::vector<int>> &image);
    void generateGauss(double sigma);

    int kernelSize;   
    int windowSize;
    //int l2;
    //double sigma;
    double gaussCoeff = 1/273;
    double harrisFactor = 0.04;

    std::vector<T> Dx; // Difference Window
    std::vector<T> Sx; // Smoothing Window
    std::vector<std::vector<T>> harrisScore;
    // Sobel_X looks like the following
    // 1    2   0   -2  -1
    // 4    8   0   -8  -4
    // 6    12  0   -12 -6
    // 4    8   0   -8  -4
    // 1    2   0   -2  -1
    std::vector<std::vector<T>> sobel_x;
    // Sobel_Y looks like the following
    // 1    4   6   4   1
    // 2    8   12  8   2
    // 0    0   0   0   0
    // -2   -8  -12 -8  -2
    // -1   -4  -6  -4  -1
    std::vector<std::vector<T>> sobel_y;
    std::vector<T> sobel_x_h;
    std::vector<T> sobel_x_v;
    std::vector<T> sobel_y_h;
    std::vector<T> sobel_y_v;
    std::vector<std::vector<T>> Ix;
    std::vector<std::vector<T>> Iy;
    std::vector<std::vector<T>> Ixy;
    std::vector<std::vector<T>> gIxx;
    std::vector<std::vector<T>> gIyy;
    std::vector<std::vector<T>> gIxy;
    std::vector<std::vector<T>> Gauss;
    std::vector<T> Gauss_h;
    std::vector<T> Gauss_v;
    //std::vector<std::vector<double>> Gauss = {{1,  4,  7,  4,  1},
    //                                          {4,  16, 26, 16, 4},
    //                                          {7,  26, 41, 26, 7},
    //                                          {4,  16, 26, 16, 4},
    //                                          {1,  4,  7,  4,  1}};
};

template <typename T>
void print(std::vector<std::vector<T>> &matrix);
template <typename T>
void print(std::vector<T> &matrix);

#endif