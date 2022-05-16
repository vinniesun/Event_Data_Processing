// g++ -std=c++11 -w -O2 -I /usr/local/Cellar/opencv/4.5.5_1/include/opencv4/ -o customharris customharris.cpp luvharris.cpp -lopencv_core -lopencv_imgcodecs -lopencv_imgproc -lopencv_highgui -lopencv_ml -lopencv_video -lopencv_features2d -lopencv_videoio
#include "customharris.hpp"
#include "luvharris.hpp"

template <typename T>
HarrisCornerDetector<T>::HarrisCornerDetector(int kSize, int wSize, int height, int width) {
    kernelSize = kSize;
    windowSize = wSize;

    Sx = std::vector<T> (kernelSize, 0);
    Dx = std::vector<T> (kernelSize, 0);
    sobel_x = std::vector<std::vector<T>> (kernelSize, std::vector<T>(kernelSize, 0));
    sobel_y = std::vector<std::vector<T>> (kernelSize, std::vector<T>(kernelSize, 0));
    sobel_x_h = std::vector<T> (kernelSize, 0);
    sobel_x_v = std::vector<T> (kernelSize, 0);
    sobel_y_h = std::vector<T> (kernelSize, 0);
    sobel_y_v = std::vector<T> (kernelSize, 0);
    harrisScore = std::vector<std::vector<T>> (height, std::vector<T>(width, 0));
    Ix = std::vector<std::vector<T>> (height, std::vector<T>(width, 0));
    Iy = std::vector<std::vector<T>> (height, std::vector<T>(width, 0));
    Ixy = std::vector<std::vector<T>> (height, std::vector<T>(width, 0));
    gIxx = std::vector<std::vector<T>> (height, std::vector<T>(width, 0));
    gIyy = std::vector<std::vector<T>> (height, std::vector<T>(width, 0));
    gIxy = std::vector<std::vector<T>> (height, std::vector<T>(width, 0));
    Gauss = std::vector<std::vector<T>> (kernelSize, std::vector<T>(kernelSize, 0));
    Gauss_h = std::vector<T> (kernelSize, 0);
    Gauss_v = std::vector<T> (kernelSize, 0);

    for (int i = 0; i < kernelSize; i++) {
        Sx[i] = factorial(kernelSize - 1)/(factorial(kernelSize - 1 - i) * factorial(i));
        Dx[i] = pascal(i, kernelSize - 2) - pascal(i-1, kernelSize - 2);
    }
    
    generateSobel(); // Create the Sobel Operator;
    generateGauss(1);
}

template <typename T>
HarrisCornerDetector<T>::~HarrisCornerDetector() {

}

template <typename T>
int HarrisCornerDetector<T>::pascal(int k, int n) {
    if (k >= 0 && k <= n) return factorial(n)/(factorial(n-k)*factorial(k));
    else return 0;
}

template <typename T>
int HarrisCornerDetector<T>::factorial(int n) {
    if (n > 1) return n * factorial(n-1);
    else return 1;
}

template <typename T>
void HarrisCornerDetector<T>::square(std::vector<std::vector<T>> &input) {
    if (input.size() != 0 && input[0].size() != 0) {
        for (auto &row: input) {
            for (auto &col: row) {
                col = col * col;
            }
        }
    }
}

template <typename T>
void HarrisCornerDetector<T>::multiply(const std::vector<std::vector<T>> &input1, const std::vector<std::vector<T>> &input2, std::vector<std::vector<T>> &output) {
    // Check if input and output dimension matches
    if (input1.size() == input2.size() && input1.size() == output.size() &&
        input1[0].size() == input2[0].size() && input1[0].size() == output[0].size()) {
        int row = input1.size();
        int col = input1[0].size();

        // elementwise multiplication
        for (int r = 0; r < row; r++) {
            for (int c = 0; c < col; c++) {
                output[r][c] = input1[r][c] * input2[r][c];
            }
        }
    }
}

template <typename T>
void HarrisCornerDetector<T>::multFactor(std::vector<std::vector<T>> &input) {
    if (input.size() != 0 && input[0].size() != 0) {
        for (auto &row: input) {
            for (auto &col: row) {
                col = col * harrisFactor;
            }
        }
    }
}

template <typename T>
void HarrisCornerDetector<T>::subtract(const std::vector<std::vector<T>> &input1, const std::vector<std::vector<T>> &input2, std::vector<std::vector<T>> &output) {
    if (input1.size() == input2.size() && input1.size() == output.size() &&
        input1[0].size() == input2[0].size() && input1[0].size() == output[0].size()) {
        int row = input1.size();
        int col = input1[0].size();

        // elementwise multiplication
        for (int r = 0; r < row; r++) {
            for (int c = 0; c < col; c++) {
                output[r][c] = input1[r][c] - input2[r][c];
            }
        }
    }
}

template <typename T>
void HarrisCornerDetector<T>::add(const std::vector<std::vector<T>> &input1, const std::vector<std::vector<T>> &input2, std::vector<std::vector<T>> &output) {
    if (input1.size() == input2.size() && input1.size() == output.size() &&
        input1[0].size() == input2[0].size() && input1[0].size() == output[0].size()) {
        int row = input1.size();
        int col = input1[0].size();

        // elementwise multiplication
        for (int r = 0; r < row; r++) {
            for (int c = 0; c < col; c++) {
                output[r][c] = input1[r][c] + input2[r][c];
            }
        }
    }
}

template <typename T>
void HarrisCornerDetector<T>::Conv2D(const std::vector<std::vector<int>> &input, std::vector<std::vector<T>> &output, const std::vector<std::vector<T>> &kernel, std::string mode) {
    double factor = 1.0;
    //if (mode == "Gauss") factor = gaussCoeff;
    //else factor = 1;

    int inputRow = input.size();
    int inputCol = input[0].size();
    int kernelRow = kernel.size();
    int kernelCol = kernel[0].size();
    int kernelCenterX = kernelCol/2;
    int kernelCenterY = kernelRow/2;
    double sum;

    // Start Timing Here
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < inputRow; i++) {
        for (int j = 0; j < inputCol; j++) {
            sum = 0;
            for (int kRow = 0; kRow < kernelRow; kRow++) {
                for (int kCol = 0; kCol < kernelCol; kCol++) {
                    int x = j - kernelCenterX + kCol;
                    int y = i - kernelCenterY + kRow;
                    if (x >= 0 && x < inputCol && y >= 0 && y < inputRow)
                        sum += input[y][x] * kernel[kRow][kCol];
                }
            }
            output[i][j] = (factor * sum) * -1;
        }
    }
    // Finish TIming Here
    auto stop = std::chrono::high_resolution_clock::now();
    auto time_taken = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    //std::cout << "Time taken for Traditional Convolution is: " << time_taken.count() << "us." << std::endl;
    std::ofstream outfile;
    outfile.open("Time_Taken.txt", std::ios_base::app);
    if (!outfile) {
        std::cout << "Failed to open Time_Taken.txt" << std::endl;
        return;
    }
    outfile << time_taken.count() << "\n";
    outfile.close();
}

template <typename T>
void HarrisCornerDetector<T>::Conv2D(const std::vector<std::vector<T>> &input, std::vector<std::vector<T>> &output, const std::vector<std::vector<T>> &kernel, std::string mode) {
    double factor = 1.0;
    //if (mode == "Gauss") factor = gaussCoeff;
    //else factor = 1;

    int inputRow = input.size();
    int inputCol = input[0].size();
    int kernelRow = kernel.size();
    int kernelCol = kernel[0].size();
    int kernelCenterX = kernelCol/2;
    int kernelCenterY = kernelRow/2;
    int sum;

    // Start Timing Here
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < inputRow; i++) {
        for (int j = 0; j < inputCol; j++) {
            sum = 0;
            for (int kRow = 0; kRow < kernelRow; kRow++) {
                for (int kCol = 0; kCol < kernelCol; kCol++) {
                    int x = j - kernelCenterX + kCol;
                    int y = i - kernelCenterY + kRow;
                    if (x >= 0 && x < inputCol && y >= 0 && y < inputRow)
                        sum += input[y][x] * kernel[kRow][kCol];
                }
            }
            output[i][j] = (factor * sum) * -1;
        }
    }
    // Finish Timing Here
    auto stop = std::chrono::high_resolution_clock::now();
    auto time_taken = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    //std::cout << "Time taken for Traditional Convolution is: " << time_taken.count() << "us." << std::endl;
    std::ofstream outfile;
    outfile.open("Time_Taken.txt", std::ios_base::app);
    if (!outfile) {
        std::cout << "Failed to open Time_Taken.txt" << std::endl;
        return;
    }
    outfile << time_taken.count() << "\n";
    outfile.close();
}

template <typename T>
void HarrisCornerDetector<T>::separableConv2D(const std::vector<std::vector<int>> &input, std::vector<std::vector<T>> &output, const std::vector<T> &kernel_h, const std::vector<T> &kernel_v, std::string mode) {
    double factor = 1.0;
    //if (mode == "Gauss") factor = gaussCoeff;
    //else factor = 1;

    int inputRow = input.size();
    int inputCol = input[0].size();
    int kernelRow = kernel_v.size();
    int kernelCol = kernel_h.size();
    int kernelCenterX = kernelCol/2;
    int kernelCenterY = kernelRow/2;
    int sum;

    // Start Timing Here
    auto start = std::chrono::high_resolution_clock::now();

    std::vector<std::vector<int>> intermediate(input);

    // Vertical First
    for (int i = 0; i < inputRow; i++) {
        for (int j = 0; j < inputCol; j++) {
            sum = 0;
            for (int kRow = 0; kRow < kernelRow; kRow++) {
                int y = i - kernelCenterY + kRow;
                if (y >= 0 && y < inputRow)
                    sum += input[y][j] * kernel_v[kRow];
            }
            intermediate[i][j] = (factor * sum);
        }
    }

    // Horizontal First
    for (int i = 0; i < inputRow; i++) {
        for (int j = 0; j < inputCol; j++) {
            sum = 0;
            for (int kCol = 0; kCol < kernelCol; kCol++) {
                int x = j - kernelCenterX + kCol;
                if (x >= 0 && x < inputCol)
                    sum += intermediate[i][x] * kernel_h[kCol];
            }
            output[i][j] = (factor * sum) * -1;
        }
    }
    // Finish Timing Here
    auto stop = std::chrono::high_resolution_clock::now();
    auto time_taken = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    //std::cout << "Time taken for Traditional Convolution is: " << time_taken.count() << "us." << std::endl;
    std::ofstream outfile;
    outfile.open("Time_Taken.txt", std::ios_base::app);
    if (!outfile) {
        std::cout << "Failed to open Time_Taken.txt" << std::endl;
        return;
    }
    outfile << time_taken.count() << "\n";
    outfile.close();
}

template <typename T>
void HarrisCornerDetector<T>::separableConv2D(const std::vector<std::vector<T>> &input, std::vector<std::vector<T>> &output, const std::vector<T> &kernel_h, const std::vector<T> &kernel_v, std::string mode) {
    double factor = 1.0;
    //if (mode == "Gauss") factor = gaussCoeff;
    //else factor = 1;

    int inputRow = input.size();
    int inputCol = input[0].size();
    int kernelRow = kernel_v.size();
    int kernelCol = kernel_h.size();
    int kernelCenterX = kernelCol/2;
    int kernelCenterY = kernelRow/2;
    int sum;

    // Start Timing Here
    auto start = std::chrono::high_resolution_clock::now();

    std::vector<std::vector<T>> intermediate(input);

    // Vertical First
    for (int i = 0; i < inputRow; i++) {
        for (int j = 0; j < inputCol; j++) {
            sum = 0;
            for (int kRow = 0; kRow < kernelRow; kRow++) {
                int y = i - kernelCenterY + kRow;
                if (y >= 0 && y < inputRow)
                    sum += intermediate[y][j] * kernel_v[kRow];
            }
            intermediate[i][j] = (factor * sum);
        }
    }

    // Horizontal First
    for (int i = 0; i < inputRow; i++) {
        for (int j = 0; j < inputCol; j++) {
            sum = 0;
            for (int kCol = 0; kCol < kernelCol; kCol++) {
                int x = j - kernelCenterX + kCol;
                if (x >= 0 && x < inputCol)
                    sum += input[i][x] * kernel_h[kCol];
            }
            output[i][j] = (factor * sum) * -1;
        }
    }
    // Finish Timing Here
    auto stop = std::chrono::high_resolution_clock::now();
    auto time_taken = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    //std::cout << "Time taken for Traditional Convolution is: " << time_taken.count() << "us." << std::endl;
    std::ofstream outfile;
    outfile.open("Time_Taken.txt", std::ios_base::app);
    if (!outfile) {
        std::cout << "Failed to open Time_Taken.txt" << std::endl;
        return;
    }
    outfile << time_taken.count() << "\n";
    outfile.close();
}

template <typename T>
void HarrisCornerDetector<T>::generateSobel() {
    // Generate Sobel_X (Vertical)
    for (int win_x = 0; win_x < kernelSize; win_x++) {
        for (int win_y = 0; win_y < kernelSize; win_y++) {
            sobel_x[win_x][win_y] = Sx[win_x]*Dx[win_y];
            if (win_y == 0) sobel_x_v[win_x] = Sx[win_x]*Dx[win_y];
            if (win_x == 0) sobel_x_h[win_y] = Sx[win_x]*Dx[win_y];
        }
    }
    // Generate Sobel_Y (Horizontal)
    for (int win_x = 0; win_x < kernelSize; win_x++) {
        for (int win_y = 0; win_y < kernelSize; win_y++) {
            sobel_y[win_x][win_y] = Sx[win_y]*Dx[win_x];
            if (win_y == 0) sobel_y_h[win_x] = Sx[win_x]*Dx[win_y];
            if (win_x == 0) sobel_y_v[win_y] = Sx[win_x]*Dx[win_y];
        }
    }
}

template <typename T>
void HarrisCornerDetector<T>::generateHarrisScore(const std::vector<std::vector<int>> &image) {
    // Generate Ixy, Ixx and Iyy
    //Conv2D(image, Ix, sobel_x, "");
    //Conv2D(image, Iy, sobel_y, "");
    separableConv2D(image, Ix, sobel_x_h, sobel_x_v, "");
    separableConv2D(image, Iy, sobel_y_h, sobel_y_v, "");

    multiply(Ix, Iy, Ixy);  // Ixy = Ix * Iy
    square(Ix);             // Ixx = Ix^2
    square(Iy);             // Iyy = Iy^2

    // Apply Gaussian Filter
    //Conv2D(Ix, gIxx, Gauss, "Gauss");   // gIxx = Ixx * g(x)
    //Conv2D(Iy, gIyy, Gauss, "Gauss");   // gIyy = Iyy * g(x)
    //Conv2D(Ixy, gIxy, Gauss, "Gauss");  // gIxy = Ixy * g(x)
    separableConv2D(Ix, gIxx, Gauss_h, Gauss_v, "Gauss");   // gIxx = Ixx * g(x)
    separableConv2D(Iy, gIyy, Gauss_h, Gauss_v, "Gauss");   // gIyy = Iyy * g(x)
    separableConv2D(Ixy, gIxy, Gauss_h, Gauss_v, "Gauss");  // gIxy = Ixy * g(x)

    // score = det(M) - k*trace(M)^2
    // Can calculate as the entire matrix or use window sum
    std::vector<std::vector<T>> det (Ix.size(), std::vector<T>(Ix[0].size(), 0));
    multiply(gIxx, gIyy, det);
    square(gIxy);
    subtract(det, gIxy, det);
    //std::cout << "determinant" << std::endl;
    //print<double>(det);
    //std::cout << "----------" << std::endl;
    std::vector<std::vector<T>> trace (Ix.size(), std::vector<T>(Ix[0].size(), 0));
    add(gIxx, gIyy, trace);
    square(trace);
    multFactor(trace);
    //std::cout << "trace" << std::endl;
    //print<double>(trace);
    //std::cout << "----------" << std::endl;
    subtract(det, trace, harrisScore);
}

/*void HarrisCornerDetector::generateGauss(double sigma) {
    double coeff = 2.0*M_PI*sigma*sigma;
    double exp_coeff = 2.0*sigma*sigma;
    double sum = 0.0; // This is used for normalisation
    double temp = 0.0;
    int kSize = (2*windowSize + 2-kernelSize)/2;

    Gauss = std::vector<std::vector<double>> (2*kSize+1, std::vector<double>(2*kSize+1, 0));

    for (int x = -kSize; x <= kSize; x++) { 
        std::cout << "temp" << std::endl;
        for (int y = -kSize; y <= kSize; y++) {
            temp = x*x + y*y;
            std::cout << temp << " ";
            Gauss[y+kSize][x+kSize] = exp(-temp/exp_coeff)/coeff;
            std::cout << Gauss[y+kSize][x+kSize] << " ";
            sum += Gauss[y+kSize][x+kSize];
        }
        std::cout << std::endl;
    }
    std::cout << sum << std::endl;
    //for (auto &row: Gauss) {
    //    for (auto &col: row) {
    //        col /= sum;
    //    }
    //}
    std::cout << "Gauss" << std::endl;
    print<double>(Gauss);
}*/
template <typename T>
void HarrisCornerDetector<T>::generateGauss(double sigma) {
    double coeff = 2.0*M_PI*sigma*sigma;
    double exp_coeff = 2.0*sigma*sigma;
    double sum = 0.0; // This is used for normalisation
    double temp = 0.0;
    int kSize = kernelSize/2;

    for (int x = -kSize; x <= kSize; x++) { 
        //std::cout << "temp" << std::endl;
        for (int y = -kSize; y <= kSize; y++) {
            temp = x*x + y*y;
            //std::cout << temp << " ";
            Gauss[y+kSize][x+kSize] = exp(-temp/exp_coeff)/coeff;
            //std::cout << Gauss[y+kSize][x+kSize] << " ";
            sum += Gauss[y+kSize][x+kSize];

            if (y == 0) Gauss_v[x] = Gauss[y+kSize][x+kSize];
            if (x == 0) Gauss_h[y] = Gauss[y+kSize][x+kSize];
        }
        //std::cout << std::endl;
    }
    //std::cout << sum << std::endl;
    //for (auto &row: Gauss) {
    //    for (auto &col: row) {
    //        col /= sum;
    //    }
    //}
    //std::cout << "Gauss" << std::endl;
    //print<double>(Gauss);
}

template <typename T>
void print(std::vector<std::vector<T>> &matrix) {
    std::cout.precision(10);

    for (auto &row:matrix) {
        for (auto &col:row) {
            std::cout << col << "\t";
        }
        std::cout << std::endl;
    }
}

template <typename T>
void print(std::vector<T> &matrix) {
    std::cout.precision(10);

    for (auto &row:matrix) {
        std::cout << row << "\t";
    }
    std::cout << std::endl;
}

int main() {
    /*HarrisCornerDetector test(5, 4, 5, 10); //test(5, 4, 180, 240)

    std::vector<std::vector<int>> dummy = {{1,2,3,4,5,6,7,8,9,10},
                                           {11,12,13,14,15,16,17,18,19,20},
                                           {21,22,23,24,25,26,27,28,29,30},
                                           {31,32,33,34,35,36,37,38,39,40},
                                           {41,42,43,44,45,46,47,48,49,50},};
    
    std::cout << "Sobel_X:" << std::endl;
    print<double>(test.sobel_x);
    std::cout << "----------------------------------------------------------" << std::endl;

    std::cout << "Sobel_Y:" << std::endl;
    print<double>(test.sobel_y);
    std::cout << "----------------------------------------------------------" << std::endl;

    std::cout << "Gauss:" << std::endl;
    print<double>(test.Gauss);
    std::cout << "----------------------------------------------------------" << std::endl;

    test.generateHarrisScore(dummy);

    std::cout << "Ixy:" << std::endl;
    print<double>(test.Ixy);
    std::cout << "----------------------------------------------------------" << std::endl;

    std::cout << "Ixx:" << std::endl;
    print<double>(test.Ix);
    std::cout << "----------------------------------------------------------" << std::endl;

    std::cout << "Iyy:" << std::endl;
    print<double>(test.Iy);
    std::cout << "----------------------------------------------------------" << std::endl;

    std::cout << "gIxx:" << std::endl;
    print<double>(test.gIxx);
    std::cout << "----------------------------------------------------------" << std::endl;

    std::cout << "Harris Response:" << std::endl;
    print<double>(test.harrisScore);
    std::cout << "----------------------------------------------------------" << std::endl;*/

    HarrisCornerDetector<double> test(7, 4, 180, 240);
    /*HarrisCornerDetector test(5, 4, 5, 10);

    print<double>(test.sobel_x_h);
    print<double>(test.sobel_x_v);
    print<double>(test.sobel_y_h);
    print<double>(test.sobel_y_v);
    print<double>(test.Gauss_h);
    print<double>(test.Gauss_v);

    std::vector<std::vector<int>> dummy = {{1,2,3,4,5,6,7,8,9,10},
                                           {11,12,13,14,15,16,17,18,19,20},
                                           {21,22,23,24,25,26,27,28,29,30},
                                           {31,32,33,34,35,36,37,38,39,40},
                                           {41,42,43,44,45,46,47,48,49,50},};

    test.generateHarrisScore(dummy);

    print<double>(test.Ix);
    print<double>(test.Iy);
    print<double>(test.Ixy);*/

    cv::VideoCapture cap("TOS.avi");

    if (!cap.isOpened()) return -1; // Failed to open video file

    cv::Mat frame, gray;
    std::vector<std::vector<int>> curFrame (180, std::vector<int>(240, 0));

    int count = 0;

    while (1) {
        cap >> frame;

        if (frame.empty() || count > 1000) break;

        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

        for (int i=0; i < gray.rows; i++) {
            for (int j=0; j < gray.cols; j++) {
                curFrame[i][j] = gray.at<uchar>(i, j);
            }
        }

        test.generateHarrisScore(curFrame);
        //std::cout << "-----------------------------" << std::endl;
        count++;
    }

    cap.release();

    return 0;
}