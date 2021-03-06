// g++ -std=c++11 -w -O2 -I /usr/local/Cellar/opencv/4.5.5_1/include/opencv4/ -o luvharris luvharris.cpp -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_ml -lopencv_video -lopencv_features2d
// clang++ -std=c++11 -w -O2 -I /usr/local/Cellar/opencv/4.5.5_1/include/opencv4/ -o luvharris luvharris.cpp -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_ml -lopencv_video -lopencv_features2d
// Everytime brew updates OpenCV, need to relink the libraries with the command 
// brew link --overwrite opencv
#include <iostream>
#include <vector>
#include "luvharris.hpp"

/*
 * This function converts a 2D vector to OpenCV's cv::Mat.
 */
cv::Mat convertVectorToMat(std::vector<std::vector<int>> &TOS) {
    cv::Mat lookup(TOS.size(), TOS[0].size(), CV_8UC1); // CV_8UC1 means 8-bits, Unsigned integer, 1 Channel
    //std::cout << lookup.rows << std::endl;
    //std::cout << lookup.cols << std::endl;
    for (int row = 0; row < lookup.rows; row++) {
        for (int col = 0; col < lookup.cols; col++) {
            //std::cout << row << "," << col << " ";
            lookup.at<int>(row, col) = TOS[row][col];
        }
        //std::cout << std::endl;
    }
    return lookup;
}

/*
 * This function generate the Harris Lookup Table by calculating the Harris Score.
 */
void generateLookup(cv::Mat &input_TOS, cv::Mat &output, int blockSize, int apertureSize, double k) {
    //cv::Mat lookup = cv::Mat::zeros(input_TOS.size(), CV_32FC1);
    //cv::cornerHarris(input_TOS, lookup, blockSize, apertureSize, k);
    cv::Mat temp;
    cv::cornerHarris(input_TOS, temp, blockSize, apertureSize, k);

    //cv::Mat lookup_norm;
    cv::normalize( temp, output, 0, 255, cv::NORM_MINMAX, CV_8UC1, cv::Mat() );

    //cv::Mat lookup_norm_scaled;
    //cv::convertScaleAbs( lookup_norm, lookup_norm_scaled );

    //return lookup_norm_scaled;
    //return lookup_norm;
}

/*
 * This function check whether the current event qualifies as a corner
 */
bool luvHarris(cv::Mat &harrisLookup, int x, int y, int t, bool p, int threshold) {
    return ((int)harrisLookup.at<int>(y, x) > threshold);
}

/*
int main() {
    std::vector<std::vector<int>> test {{0, 200, 255, 200, 0,     0,   0},
                                        {0, 200, 255, 200, 0,     0,   0},
                                        {0, 200, 255, 200, 0,     0,   0},
                                        {0, 200, 255, 200, 200,   200, 200},
                                        {0, 200, 255, 255, 255,   255, 255},
                                        {0, 200, 200, 200, 200,   200, 200},
                                        {0, 0,   0,   0,   0,     0,   0}};

    std::vector<std::vector<int>> test_abs  {{0, -200, -255, -200, 0,     0,   0},
                                             {0, -200, -255, -200, 0,     0,   0},
                                             {0, -200, -255, -200, 0,     0,   0},
                                             {0, -200, -255, -200, -200,   -200, -200},
                                             {0, -200, -255, -255, -255,   -255, -255},
                                             {0, -200, -200, -200, -200,   -200, -200},
                                             {0, 0,   0,   0,   0,     0,   0}};

    std::cout << "Original Matrix" << std::endl;
    for (auto row:test) {
        for (auto col:row) {
            std::cout << col << "\t";
        }
        std::cout << std::endl;
    }

    cv::Mat result = convertVectorToMat(test);

    //std::cout << result.rows << std::endl;
    //std::cout << result.cols << std::endl;

    int blockSize=2;
    int apertureSize=3;
    double k=0.04;

    cv::Mat harrisLookup = generateLookup(result, blockSize, apertureSize, k);
    std::cout << "Normalised Harris Score Matrix" << std::endl;
    for (int row = 0; row < harrisLookup.rows; row++) {
        for (int col = 0; col < harrisLookup.cols; col++) {
            // To print values from OpenCV's Mat, we need to typecast it
            // Note. The normalised matrix is in float, so to use its value,
            // we need to typecast it to int first.
            std::cout << (int)harrisLookup.at<float>(row, col) << "\t"; 
        }
        std::cout << std::endl;
    }
    
    std::cout << "Corner Check" << std::endl;
    for (int row = 0; row < harrisLookup.rows; row++) {
        for (int col = 0; col < harrisLookup.cols; col++) {
            std::cout << luvHarris(harrisLookup, col, row, 0, 0, 200) << "\t"; // To print values from OpenCV's Mat, we need to typecast it
        }
        std::cout << std::endl;
    }

}*/