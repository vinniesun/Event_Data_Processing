// g++ -std=c++11 -w -O2 -I /usr/local/Cellar/eigen/3.4.0_1/include/eigen3/ -I /usr/local/Cellar/opencv/4.5.5_1/include/opencv4/ -o main process.cpp efast.cpp arcstar.cpp luvharris.cpp main.cpp -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_ml -lopencv_video -lopencv_features2d
#include "process.hpp"
#include "efast.hpp"
#include "arcstar.hpp"
#include "luvharris.hpp"
#include "customharris.hpp"
#include "progressbar.hpp"

#include <thread>
#include <atomic>
#include <mutex>

std::string output = "./Output/";
std::string tos_location = "TOS/";
std::string luvharris_result = "luvharris/";
std::string patch_result = "patch/";

std::string input_file = "/Users/vincent/Desktop/CityUHK/Event_Process/Event_Camera_Dataset/shapes_rotation/events.txt";
//std::string input_file = "/Users/vincent/Desktop/CityUHK/Event_Process/Event_Camera_Dataset/shapes_translation/events.txt";
//std::string input_file = "/Users/vincent/Desktop/CityUHK/Event_Process/Event_Camera_Dataset/shapes_6dof/events.txt";
//std::string input_file = "/Users/vincent/Desktop/CityUHK/Event_Process/Event_Camera_Dataset/poster_rotation/events.txt";
//std::string input_file = "/Users/vincent/Desktop/CityUHK/Event_Process/Event_Camera_Dataset/poster_6dof/events.txt";

void harrisCalc(cv::Mat &sae_1, cv::Mat &harris_response, int blockSize, int apertureSize, double k, int &thread_count, std::atomic_bool &work_status) {
    while (work_status) {
        cv::cornerHarris(sae_1, harris_response, blockSize, apertureSize, k);
        thread_count++;
    }
    std::cout << "Thread Ended" << std::endl;
}

int main() {
    double threshold = 0.0;

    int ref_period = 1000;
    int nn_window = 5000;

    Events current_event = read_file(input_file);

    int max_x = current_event.width-1;
    int max_y = current_event.height-1;

    Events ref_event;
    refractoryFiltering(current_event, ref_event, ref_period);
    ref_event.event_count = ref_event.x.size();
    std::cout << "Number of events after ref filtering: " << ref_event.event_count << std::endl;

    current_event.~Events();
    
    Events nn_event;

    nnFiltering(ref_event, nn_event, nn_window);
    nn_event.event_count = nn_event.x.size();
    std::cout << "Number of Events after NN filtering: " << nn_event.event_count << std::endl;

    ref_event.~Events();

    //std::vector<std::vector<int>> sae_1 (current_event.height, std::vector<int>(current_event.width, 0));
    //std::vector<std::vector<int>> sae_0 (current_event.height, std::vector<int>(current_event.width, 0));
    //cv::Mat sae_1 = cv::Mat::zeros(current_event.height, current_event.width, CV_8UC1);
    cv::Mat sae_1 = cv::Mat(current_event.height, current_event.width, CV_8UC1, cv::Scalar(0));
    cv::Mat sae_0 = cv::Mat::zeros(current_event.height, current_event.width, CV_8UC1);

    //std::cout << sae_1.size() << ", " << sae_1[0].size() << std::endl;

    int prev_time = 0;

    int efast_corner = 0;
    int arcstar_corner = 0;
    int luvharris_corner = 0;
    int patch_corner = 0;
    int custom_corner = 0;

    int cycle = 0;
    int cycle2 = 0;

    // Corner Evaluation
    std::ofstream outfile;
    int quant = (int)std::pow(2.0, 16.0); // Template is double std::pow(double base, double exponent), hence we need to cast it to an int
    int ktos = 3;
    int ttos = 2*(2*ktos + 1);
    // for original luvHarris, threshold of 0.12 seems to be ideal.
    // for patch luvHarris, threshold of 0.11 seems to be ideal.
    float mean;
    float std_dev;
    float luvharris_threshold = 0.00003902; // Default is 200 when normalised, 0.1 when it's the raw harris score
    float patch_harris_threshold = 0.11; // Default is 200 when normalised, 0.1 when it's the raw harris score
    int blockSize=2; // Neighbourhood Size. Default is 2
    int apertureSize=3; // This is the Sobel Operator's Size. Default is 3.
    double k=0.04;
    cv::Mat lookup_1 = cv::Mat::zeros(sae_1.rows, sae_1.cols, CV_8UC1);
    cv::Mat lookup_0 = cv::Mat::zeros(sae_1.rows, sae_1.cols, CV_8UC1);
    //cv::Mat patch_lookup_harris = cv::Mat::zeros(sae_1.rows, sae_1.cols, CV_32FC1);
    //cv::Mat patch_lookup_normalised = cv::Mat::zeros(sae_1.rows, sae_1.cols, CV_8UC1);

    cv::Mat harris_response(sae_1.rows, sae_1.cols, CV_32FC1);

    cv::Mat temp_sae = cv::Mat::zeros(sae_1.rows, sae_1.cols, CV_8UC1);
    cv::Mat greyscale_img(sae_1.rows, sae_1.cols, CV_8UC1);
    cv::Mat colour_img(sae_1.rows, sae_1.cols, CV_8UC3);
    cv::Mat corner_trail(sae_1.rows, sae_1.cols, CV_8UC1, cv::Scalar(0));

    //HarrisCornerDetector<float> detector(5,2,180,240, CV_32FC1);
    int cumulated_time = 0;
    int count = 0;
    int thread_count = 0;
    std::atomic_bool work_status (true);

    std::mutex mtx;

    //cv::VideoWriter video("patch_sae.avi", cv::VideoWriter::fourcc('X', 'V', 'I', 'D'), 120, cv::Size(nn_event.width, nn_event.height), false);
    //cv::VideoWriter video("patch_event_trail.avi", cv::VideoWriter::fourcc('X', 'V', 'I', 'D'), 120, cv::Size(nn_event.width, nn_event.height), true);

    std::thread thread1 (harrisCalc, std::ref(sae_1), std::ref(harris_response), blockSize, apertureSize, k, std::ref(thread_count), std::ref(work_status));

    // Start Timing Here
    auto start = std::chrono::high_resolution_clock::now();
    // For Original luvHarris Approach
    for (int i = 0; i < 4000000; i++) {
    //for (int i = 400000; i < 500000; i++) {
        progressBar(4000000, i);

        const int pol = nn_event.p[i] ? 1:0;
        int32_t x = nn_event.x[i];
        int32_t y = nn_event.y[i];
        int32_t t = nn_event.t[i];

        // original luvHarris
        mtx.lock();
        update_sae(sae_1, "TOS", x, y, t, 66000, quant, prev_time, ktos, ttos, max_x, max_y);
        mtx.unlock();
        
        //std::thread thread1 (harrisCalc, std::ref(sae_1), std::ref(harris_response), blockSize, apertureSize, k, std::ref(thread_count), std::ref(work));

        /*float total = 0.0;
        int N = harris_response.rows * harris_response.cols;
        for (int i = 0; i < harris_response.rows; i++) {
            for (int j = 0; j < harris_response.cols; j++) {
                total += harris_response.at<float>(i, j);
            }
        }
        mean = (float)(total / N);
        float variance = 0.0;
        for (int i = 0; i < harris_response.rows; i++) {
            for (int j = 0; j < harris_response.cols; j++) {
                variance += (harris_response.at<float>(i, j) - mean)*(harris_response.at<float>(i, j) - mean);
            }
        }
        variance /= N;
        std_dev = std::sqrt(variance);

        luvharris_threshold = mean + 2*std_dev;*/

        if (harris_response.at<float>(y, x) > luvharris_threshold) {
        //if (luvHarris(lookup_1, x, y, t, pol, harris_threshold)) {
            // cv::cvtColor(sae_1, colour_img, cv::COLOR_GRAY2BGR);
            // colour_img.at<cv::Vec3b>(y, x) = cv::Vec3b(255, 255, 0);
            // cv::circle(colour_img, cv::Point(x, y), 3, cv::Scalar(0,0,255));
            // cv::imwrite(output + tos_location + std::to_string(i) +".jpg", colour_img);
            luvharris_corner++;
        }

        prev_time = t;
    }

    mtx.lock();
    work_status = false;
    mtx.unlock();
    thread1.join();

    // Finish Timing Here
    auto stop = std::chrono::high_resolution_clock::now();
    auto time_taken = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    std::cout << "Time taken to Process is: " << time_taken.count() << "us." << std::endl;

    // int time_frame = nn_event.t[999999] - nn_event.t[0];
    // std::cout << "Time frame is: " << time_frame << std::endl;

    std::cout << "Number of luvHarris corner: " << luvharris_corner << std::endl;
    std::cout << "Number of time the thread executed Harris Response Calc is: " << thread_count << std::endl;

    return 0;
}