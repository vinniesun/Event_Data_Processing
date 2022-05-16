#ifndef LUVHARRIS_H
#define LUVHARRIS_H

#include "opencv2/opencv.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/core.hpp"
//#include <opencv2/core/mat.hpp>

cv::Mat convertVectorToMat(std::vector<std::vector<int>> &TOS);
void generateLookup(cv::Mat &input_TOS, cv::Mat &output, int blockSize, int apertureSize, double k);
bool luvHarris(cv::Mat &harrisLookup, int x, int y, int t, bool p, int threshold);

#endif