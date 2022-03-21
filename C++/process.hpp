#include <iostream>
#include <Eigen/Dense>
#include <vector>
#include <fstream>
#include <string>
#include <algorithm>

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