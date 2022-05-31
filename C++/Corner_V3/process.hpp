#pragma once

#ifndef PROCESS_H
#define PROCESS_H

#include <iostream>
#include <Eigen/Dense>
#include <vector>
#include <fstream>
#include <string>
#include <algorithm>
#include <cmath>
#include <chrono>

#include "opencv2/opencv.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/core.hpp"

class Events
{
public:
    Events(int size);
    Events();
    ~Events();

    std::vector<int32_t> t;
    std::vector<int32_t> x;
    std::vector<int32_t> y;
    std::vector<bool> p;
    int event_count;
    static const int height = 180;
    static const int width = 240;
};

Events read_file(std::string input_file);
void update_sae(cv::Mat &sae, std::string mode, int32_t x, int32_t y, int32_t t, int deltaMax, int quant, int prev_time, int ktos, int ttos, int maxX, int maxY);
void refractoryFiltering(Events &currentEvent, Events &outputEvent, int ref_period);
void nnFiltering(Events &currentEvent, Events &outputEvent, int nn_window);

#endif