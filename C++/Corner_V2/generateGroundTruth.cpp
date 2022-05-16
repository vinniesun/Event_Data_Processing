// g++ -std=c++11 -w -O2 -I /usr/local/Cellar/eigen/3.4.0_1/include/eigen3/ -I /usr/local/Cellar/opencv/4.5.5_1/include/opencv4/ -o generateVideo generateVideo.cpp -lopencv_core -lopencv_imgcodecs -lopencv_imgproc -lopencv_highgui -lopencv_ml -lopencv_video -lopencv_features2d -lopencv_videoio
// g++ -std=c++11 -w -O2 -I /usr/local/Cellar/eigen/3.4.0_1/include/eigen3/ -I /usr/local/Cellar/opencv/4.5.5_1/include/opencv4/ -o generateGroundTruth generateGroundTruth.cpp luvharris.cpp -lopencv_core -lopencv_imgcodecs -lopencv_imgproc -lopencv_highgui -lopencv_ml -lopencv_video -lopencv_features2d -lopencv_videoio -lopencv_photo
#include <iostream>
#include "opencv2/opencv.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/core.hpp"
#include <opencv2/highgui/highgui.hpp>  // Video write
#include <Eigen/Dense>
#include <fstream>
#include <vector>
#include "luvharris.hpp"

static float multiplier = 10.0e6;
std::string output = "./Outupt/groundTruth/";

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

Events::Events(int size) {
    t.reserve(size);
    x.reserve(size);
    y.reserve(size);
    p.reserve(size);
    event_count = size;
}

Events::Events() {

}

Events::~Events() {

}

Events read_file() {
    int lines = 0;
    std::fstream event_file;

    event_file.open("/Users/vincent/Desktop/CityUHK/Event_Process/Event_Camera_Dataset/shapes_rotation/events.txt", std::ios::in);
    //event_file.open("/Users/vincent/Desktop/CityUHK/Event_Process/Event_Camera_Dataset/shapes_translation/events.txt", std::ios::in);
    //event_file.open("/Users/vincent/Desktop/CityUHK/Event_Process/Event_Camera_Dataset/shapes_6dof/events.txt", std::ios::in);
    //event_file.open("/Users/vincent/Desktop/CityUHK/Event_Process/Event_Camera_Dataset/poster_rotation/events.txt", std::ios::in);
    
    if (event_file.is_open()) {
        std::string temp;
        while (std::getline(event_file, temp)) lines++;
    }
    event_file.close();

    std::cout << "number of lines found: " << lines << std::endl;

    Events all_events(lines);

    event_file.open("/Users/vincent/Desktop/CityUHK/Event_Process/Event_Camera_Dataset/shapes_rotation/events.txt", std::ios::in);
    //event_file.open("/Users/vincent/Desktop/CityUHK/Event_Process/Event_Camera_Dataset/shapes_translation/events.txt", std::ios::in);
    //event_file.open("/Users/vincent/Desktop/CityUHK/Event_Process/Event_Camera_Dataset/shapes_6dof/events.txt", std::ios::in);
    //event_file.open("/Users/vincent/Desktop/CityUHK/Event_Process/Event_Camera_Dataset/poster_rotation/events.txt", std::ios::in);
    lines = 0;
    if (event_file.is_open()) {
        std::string temp;
        while (std::getline(event_file, temp)) {
            std::stringstream ss(temp);
            std::string s;
            int count = 0;
            while (ss.good()) {
                std::getline(ss, s, ' ');
                if (count == 0) {
                    float f = std::stof(s);
                    int time = (int)(f*multiplier);
                    all_events.t[lines] = time;
                } else if (count == 1) {
                    int x = std::stoi(s);
                    all_events.x[lines] = x;
                } else if (count == 2) {
                    int y = std::stoi(s);
                    all_events.y[lines] = y;
                } else {
                    s.erase(std::remove(s.begin(), s.end(), '\n'), s.end());
                    if (s == "1") all_events.p[lines] = true;
                    else all_events.p[lines] = false;
                }
                count++;
            }
            lines++;
        }
    }
    event_file.close();
    std::cout << "finished processing" << std::endl;

    return all_events;
}

void update_sae(cv::Mat &sae, std::string mode, int32_t x, int32_t y, int32_t t, int deltaMax=66000, int quant=0, int prev_time=0, int ktos=0, int ttos=0, int maxX=0, int maxY=0) {
    if (mode == "timestamp") {
        sae.at<int>(y, x) = t;
    }
    /*else if (mode == "delta") {
        int deltaT = t - prev_time;

        for (auto& row:sae) {
            for (auto& col:row) {
                col = ((col - deltaT) > 0) ? (col - deltaT) : 0;
            }
        }
        sae.at<int>(y, x) = deltaMax;
    }
    else if (mode == "factor") {
        sae.at<int>(y, x) = (t/quant);
    }
    else if (mode == "delta_factor") {
        int deltaT = t - prev_time;

        for (auto& row:sae) {
            for (auto& col:row) {
                col = ((col - deltaT) > 0) ? (((col - deltaT)/4)*4) : 0;
            }
        }
        sae.at<int>(y, x) = deltaMax*4;
    }*/
    else if (mode == "TOS") {
        if (!(x < ktos || x > maxX - ktos || y < ktos || y > maxY - ktos)) {
            for (int i = x - ktos; i <= x + ktos; i++) { //Col
                for (int j = y - ktos; j <= y + ktos; j++) { //Row
                    sae.at<unsigned char>(j, i) = ((sae.at<unsigned char>(j, i) - 1) > (255 - ttos)) ? (sae.at<unsigned char>(j, i) - 1) : 0;
                }
            }
            sae.at<unsigned char>(y, x) = 255;
        }   
    }
}

int main() {
    int ref_period = 1000;
    int nn_window = 5000;

    Events current_event = read_file();

    Events ref_event;
    Eigen::MatrixXd ref_mask = Eigen::MatrixXd::Constant(current_event.height, current_event.width, -ref_period);

    // Refractory Filtering
    for (int i = 0; i < current_event.event_count; i++) {
        if (current_event.t[i] - ref_mask(current_event.y[i], current_event.x[i]) > ref_period) {
            ref_event.x.push_back(current_event.x[i]);
            ref_event.y.push_back(current_event.y[i]);
            ref_event.t.push_back(current_event.t[i]);
            ref_event.p.push_back(current_event.p[i]);
            ref_mask(current_event.y[i], current_event.x[i]) = current_event.t[i];
        }
    }
    ref_mask.resize(0,0);
    ref_event.event_count = ref_event.x.size();
    std::cout << "Number of events after ref filtering: " << ref_event.event_count << std::endl;

    int max_x = current_event.width-1;
    int max_y = current_event.height-1;
    int x_prev = 0;
    int y_prev = 0;
    int p_prev = 0;
    Events nn_event;
    //mask = Eigen::MatrixXd::Constant(current_event.height, current_event.width, -nn_window);
    std::vector<std::vector<int>> nn_mask(current_event.height, std::vector<int>(current_event.width, -nn_window));

    // Nearest Neighbourhood Filtering
    for (int i = 0; i < ref_event.event_count; i++) {
        int x = ref_event.x[i];
        int y = ref_event.y[i];
        int t = ref_event.t[i];
        int p = ref_event.p[i];

        if (x_prev != x || y_prev != y || p_prev != p) {
            nn_mask[y][x] = -nn_window;
            
            auto min_x_sub = std::max(0, x-1);
            auto max_x_sub = std::min(max_x, x+1);
            auto min_y_sub = std::max(0, y-1);
            auto max_y_sub = std::min(max_y, y+1);

            std::vector<int> temp;
            for (int j = min_y_sub; j < max_y_sub+1; j++) {
                for (int k = min_x_sub; k < max_x_sub+1; k++) {
                    temp.push_back(nn_mask[j][k]);
                }
            }

            for (auto& v:temp) {
                v = t - v;
            }

            int t_min = *std::min_element(temp.begin(), temp.end());
            if (t_min <= nn_window) {
                nn_event.x.push_back(x);
                nn_event.y.push_back(y);
                nn_event.t.push_back(t);
                nn_event.p.push_back(p);
            }
        }
        nn_mask[y][x] = t;
        x_prev = x;
        y_prev = y;
        p_prev = p;
    }
    nn_event.event_count = nn_event.x.size();
    std::cout << "Number of Events after NN filtering: " << nn_event.event_count << std::endl;

    cv::Mat sae_1 = cv::Mat::zeros(nn_event.height, nn_event.width, CV_8UC1);

    int prev_time = 0;
    int quant = (int)std::pow(2.0, 16.0); // Template is double std::pow(double base, double exponent), hence we need to cast it to an int
    int ktos = 3;
    int ttos = 2*(2*ktos + 1);
    int harris_threshold = 200; // Default is 200
    int blockSize=2; // Neighbourhood Size. Default is 2
    int apertureSize=3; // This is the Sobel Operator's Size. Default is 3.
    double k=0.04;
    int total_time = 0;
    //std::vector<cv::Mat> images;

    cv::VideoWriter video("./Tests/test.avi", cv::VideoWriter::fourcc('X', 'V', 'I', 'D'), 120, cv::Size(nn_event.width, nn_event.height), false);

    for (int i = 400000; i < 500000; i++) {
        const int pol = nn_event.p[i] ? 1:0;
        int32_t x = nn_event.x[i];
        int32_t y = nn_event.y[i];
        int32_t t = nn_event.t[i];

        update_sae(sae_1, "TOS", x, y, t, 66000, quant, prev_time, ktos, ttos, max_x, max_y);

        total_time += (t - prev_time);
        if (total_time >= 66000) {
            //cv::Mat temp;
            //generateLookup(sae_1, temp, blockSize, apertureSize, k); // This is the corner response matrix
            //video.write(temp);
            //cv::Mat temp = sae_1;
            //images.push_back(temp);
            video.write(sae_1);
            cv::imwrite("./Tests/" + std::to_string(i) + ".jpg", sae_1);
            total_time = 0;
        }
        prev_time = t;
    }

    video.release();

    //std::cout << images.size() << std::endl;
    //cv::Mat diff = images[0] != images[1];
    //bool eq = cv::countNonZero(diff) == 0;
    //std::cout <<  eq  << std::endl;
    //for (auto image: images) {
    //    video.write(image);
    //}

    std::cout << "Finished!" << std::endl;

    return 0;
}