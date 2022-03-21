// Homebrew install Eigen @ /usr/local/Cellar/eigen/3.4.0_1/include/eigen3/
// Compile Command
// g++ -std=c++11 -w -I /usr/local/Cellar/eigen/3.4.0_1/include/eigen3/ -o process process.cpp efast.cpp arcstar.cpp
// g++ -std=c++11 -w -O2 -I /usr/local/Cellar/eigen/3.4.0_1/include/eigen3/ -o process process.cpp efast.cpp arcstar.cpp
#include "process.hpp"
#include "efast.hpp"
#include "arcstar.hpp"

using namespace Eigen;

static float multiplier = 10.0e6;

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
    //event_file.open("/Users/vincent/Desktop/CityUHK/Event_Camera_Dataset/shapes_rotation/events.txt", std::ios::in);
    //event_file.open("/Users/vincent/Desktop/CityUHK/Event_Camera_Dataset/shapes_translation/events.txt", std::ios::in);
    event_file.open("/Users/vincent/Desktop/CityUHK/Event_Camera_Dataset/shapes_6dof/events.txt", std::ios::in);
    
    if (event_file.is_open()) {
        std::string temp;
        while (std::getline(event_file, temp)) lines++;
    }
    event_file.close();

    std::cout << "number of lines found: " << lines << std::endl;

    Events all_events(lines);

    //event_file.open("/Users/vincent/Desktop/CityUHK/Event_Camera_Dataset/shapes_rotation/events.txt", std::ios::in);
    //event_file.open("/Users/vincent/Desktop/CityUHK/Event_Camera_Dataset/shapes_translation/events.txt", std::ios::in);
    event_file.open("/Users/vincent/Desktop/CityUHK/Event_Camera_Dataset/shapes_6dof/events.txt", std::ios::in);
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

int main() {
    double threshold = 0.0;
    //Eigen::MatrixXd masks = Eigen::MatrixXd::Constant(180, 240, -100);
    //masks = masks.unaryExpr([threshold](double x) -> double {return (x < threshold) ? threshold : x;});
    //std::cout << masks << std::endl;

    int ref_period = 1000;
    int nn_window = 5000;

    Events current_event = read_file();
    
    //std::cout << current_event.t[0] << ", " << current_event.x[0] << ", " << current_event.y[0] << ", "  << current_event.p[0] << std::endl;
    //std::cout << current_event.t[1] << ", " << current_event.x[1] << ", " << current_event.y[1] << ", "  << current_event.p[1] << std::endl;

    Events ref_event;
    Eigen::MatrixXd ref_mask = Eigen::MatrixXd::Constant(current_event.height, current_event.width, -ref_period);
    //std::cout << mask.rows() << ", " << mask.cols() << std::endl;

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

    //Eigen::MatrixXd sae_[2];

    //sae_[0] = Eigen::MatrixXd::Zero(current_event.height, current_event.width);
    //sae_[1] = Eigen::MatrixXd::Zero(current_event.height, current_event.width);

    std::vector<std::vector<int>> sae_1 (current_event.height, std::vector<int>(current_event.width, 0));
    std::vector<std::vector<int>> sae_0 (current_event.height, std::vector<int>(current_event.width, 0));

    //std::cout << sae_1.size() << ", " << sae_1[0].size() << std::endl;

    int prev_time = 0;

    int efast_corner = 0;
    int arcstar_corner = 0;
    // delta timestamp
    /*for (int i = 400000; i < 500000; i++) {
        //const int pol = ref_event.p[i] ? 1:0;
        const int pol = nn_event.p[i] ? 1:0;
        //int32_t x = ref_event.x[i];
        //int32_t y = ref_event.y[i];
        //int32_t t = ref_event.t[i];
        int32_t x = nn_event.x[i];
        int32_t y = nn_event.y[i];
        int32_t t = nn_event.t[i];

        int deltaT = t - prev_time;
        if (pol) {
            for (auto& row:sae_1) {
                for (auto& col:row) {
                    col = ((col - deltaT) > 0) ? (col - deltaT) : 0;
                }
            }
            sae_1[y][x] = 66000;
            if (eFast(sae_1, x, y, t, pol)) efast_corner++;
            if (arcStar(sae_1, x, y, t, pol)) arcstar_corner++;
        } else {
            for (auto& row:sae_0) {
                for (auto& col:row) {
                    col = ((col - deltaT) > 0) ? (col - deltaT) : 0;
                }
            }
            sae_0[y][x] = 66000;
            if (eFast(sae_0, x, y, t, pol)) efast_corner++;
            if (arcStar(sae_0, x, y, t, pol)) arcstar_corner++;
        }
        prev_time = t;
        //sae_[pol].array() -= deltaT;
        //sae_[pol](y, x) = 66000;
        //sae_[pol] = sae_[pol].unaryExpr([threshold](double x) -> double {return (x < threshold) ? threshold : x;});

        //if (eFast(sae_, x, y, t, p)) corner++;
    }*/

    // Absolute Timestamp
    for (int i = 400000; i < 500000; i++) {
        const int pol = nn_event.p[i] ? 1:0;
        int32_t x = nn_event.x[i];
        int32_t y = nn_event.y[i];
        int32_t t = nn_event.t[i];

        if (pol) {
            sae_1[y][x] = t;
            if (eFast(sae_1, x, y, t, pol)) efast_corner++;
            if (arcStar(sae_1, x, y, t, pol)) arcstar_corner++;
        } else {
            sae_0[y][x] = t;
            if (eFast(sae_0, x, y, t, pol)) efast_corner++;
            if (arcStar(sae_0, x, y, t, pol)) arcstar_corner++;
        }
    }

    std::cout << "Number of eFast corner: " << efast_corner << std::endl;
    std::cout << "Number of arcStar corner: " << arcstar_corner << std::endl;

    return 0;
}