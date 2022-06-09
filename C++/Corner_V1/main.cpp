// g++ -std=c++11 -w -O2 -I /usr/local/Cellar/eigen/3.4.0_1/include/eigen3/ -I /usr/local/Cellar/opencv/4.5.5_1/include/opencv4/ -o main process.cpp efast.cpp arcstar.cpp luvharris.cpp main.cpp -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_ml -lopencv_video -lopencv_features2d
#include "process.hpp"
#include "efast.hpp"
#include "arcstar.hpp"
#include "luvharris.hpp"

int main() {
    double threshold = 0.0;

    int ref_period = 1000;
    int nn_window = 5000;

    Events current_event = read_file();

    Events ref_event;
    refractoryFiltering(current_event, ref_event, ref_period);
    ref_event.event_count = ref_event.x.size();
    std::cout << "Number of events after ref filtering: " << ref_event.event_count << std::endl;

    int max_x = current_event.width-1;
    int max_y = current_event.height-1;
    Events nn_event;

    nnFiltering(ref_event, nn_event, nn_window);
    nn_event.event_count = nn_event.x.size();
    std::cout << "Number of Events after NN filtering: " << nn_event.event_count << std::endl;

    // std::vector<std::vector<int>> sae_1 (current_event.height, std::vector<int>(current_event.width, 0));
    // std::vector<std::vector<int>> sae_0 (current_event.height, std::vector<int>(current_event.width, 0));

    // //std::cout << sae_1.size() << ", " << sae_1[0].size() << std::endl;

    // int prev_time = 0;

    // int efast_corner = 0;
    // int arcstar_corner = 0;
    // int luvharris_corner = 0;

    // int cycle = 0;

    // // Corner Evaluation
    // std::ofstream outfile;
    // int quant = (int)std::pow(2.0, 16.0); // Template is double std::pow(double base, double exponent), hence we need to cast it to an int
    // int ktos = 3;
    // int ttos = 2*(2*ktos + 1);
    // int harris_threshold = 240; // Default is 200
    // int blockSize=2; // Neighbourhood Size. Default is 2
    // int apertureSize=5; // This is the Sobel Operator's Size. Default is 3.
    // double k=0.04;
    // cv::Mat lookup_1 = cv::Mat::zeros(sae_1.size(), sae_1[0].size(), CV_8UC1);
    // cv::Mat lookup_0 = cv::Mat::zeros(sae_1.size(), sae_1[0].size(), CV_8UC1);

    // std::vector<std::vector<int>> local_sae_1 (ttos/2, std::vector<int>(ttos/2, 0));
    // std::vector<std::vector<int>> local_sae_0 (ttos/2, std::vector<int>(ttos/2, 0));
    // //cv::Mat lookup_1 = cv::Mat::zeros(local_sae_1.size(), local_sae_1[0].size(), CV_8UC1);
    // //cv::Mat lookup_0 = cv::Mat::zeros(local_sae_1.size(), local_sae_1[0].size(), CV_8UC1);

    // cv::Mat temp_sae;

    // // Start Timing Here
    // auto start = std::chrono::high_resolution_clock::now();
    // // For Original luvHarris Approach
    // for (int i = 400000; i < 500000; i++) {
    //     const int pol = nn_event.p[i] ? 1:0;
    //     int32_t x = nn_event.x[i];
    //     int32_t y = nn_event.y[i];
    //     int32_t t = nn_event.t[i];

    //     if (pol) {
    //         //Possible Mode Choices: "timestamp", "delta", "factor", "delta_factor", "TOS"
    //         update_sae(sae_1, "TOS", x, y, t, 66000, quant, prev_time, ktos, ttos, max_x, max_y);
    //         /*if (eFast(sae_1, x, y, t, pol)) {
    //             efast_corner++;
    //             outfile.open("eFast_Corners.txt", std::ios_base::app);
    //             if (!outfile) {
    //                 std::cout << "Failed to open eFast_Corners.txt" << std::endl;
    //                 return -1;
    //             }
    //             outfile << t << "," << x << "," << y << "," << pol << "\n";
    //             outfile.close();
    //         }
    //         if (arcStar(sae_1, x, y, t, pol)) {
    //             arcstar_corner++;
    //             outfile.open("arcStar_Corners.txt", std::ios_base::app);
    //             if (!outfile) {
    //                 std::cout << "Failed to open arcStar_Corners.txt" << std::endl;
    //                 return -1;
    //             }
    //             outfile << t << "," << x << "," << y << "," << pol << "\n";
    //             outfile.close();
    //         }*/
    //         if (cycle > 0) {
    //             //perform luvharris
    //             if (luvHarris(lookup_1, x, y, t, pol, harris_threshold)) luvharris_corner++;
    //         } else {
    //             //update harris lookup
    //             temp_sae = convertVectorToMat(sae_1);
    //             generateLookup(temp_sae, lookup_1, blockSize, apertureSize, k);
    //         }
    //     } else {
    //         update_sae(sae_0, "TOS", x, y, t, 66000, quant, prev_time, 0, 0, max_x, max_y);
    //         /*if (eFast(sae_0, x, y, t, pol)) {
    //             efast_corner++;
    //             outfile.open("eFast_Corners.txt", std::ios_base::app);
    //             if (!outfile) {
    //                 std::cout << "Failed to open eFast_Corners.txt" << std::endl;
    //                 return -1;
    //             }
    //             outfile << t << "," << x << "," << y << "," << pol << "\n";
    //             outfile.close();
    //         }
    //         if (arcStar(sae_0, x, y, t, pol)) {
    //             arcstar_corner++;
    //             outfile.open("arcStar_Corners.txt", std::ios_base::app);
    //             if (!outfile) {
    //                 std::cout << "Failed to open arcStar_Corners.txt" << std::endl;
    //                 return -1;
    //             }
    //             outfile << t << "," << x << "," << y << "," << pol << "\n";
    //             outfile.close();
    //         }*/
    //         if (cycle > 0) {
    //             //perform luvharris
    //             if (luvHarris(lookup_0, x, y, t, pol, harris_threshold)) luvharris_corner++;
    //         } else {
    //             //update harris lookup
    //             temp_sae = convertVectorToMat(sae_0);
    //             generateLookup(temp_sae, lookup_0, blockSize, apertureSize, k);
    //         }
    //     }
    //     prev_time = t;

    //     if (cycle == 5000) {
    //         cycle = 0;
    //     } else {
    //         cycle++;
    //     }

    //     outfile.open("All_events.txt", std::ios_base::app);
    //     if (!outfile) {
    //         std::cout << "Failed to open All_events.txt" << std::endl;
    //         return -1;
    //     }
    //     outfile << t << "," << x << "," << y << "," << pol << "\n";
    //     outfile.close();
    // }

    // // Finish Timing Here
    // auto stop = std::chrono::high_resolution_clock::now();
    // auto time_taken = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    // std::cout << "Time taken to Process is: " << time_taken.count() << "us." << std::endl;

    // std::cout << "Number of eFast corner: " << efast_corner << std::endl;
    // std::cout << "Number of arcStar corner: " << arcstar_corner << std::endl;
    // std::cout << "Number of luvHarris corner: " << luvharris_corner << std::endl;

    // return 0;
}