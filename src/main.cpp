#include "ImageRecogn.hpp"
#include "Segmentation.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <iostream>

int main(int argc, char *argv[]) {
    if (argc != 2) {
        std::cout << "usage: " << argv[0] << " <filename>" << std::endl;
    }
    cv::Mat image = cv::imread(argv[1]);
    cv::Mat converted = preprocess(image);
    // cv::Mat max = selectMax(image);
    // cv::imshow("Lena",image);
    cv::imshow("Converted", converted);
    cv::waitKey(-1);
    return 0;
}
