#include "ImageRecogn.cpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <iostream>

int main(int, char *[]) {
    std::cout << "Start ..." << std::endl;
    cv::Mat image = cv::imread("../data/Lena.png");
    cv::Mat max = selectMax(image);
    cv::imshow("Lena",image);
    cv::imshow("Max",max);
    cv::waitKey(-1);
    return 0;
}
