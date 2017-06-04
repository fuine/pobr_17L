#include "ImageRecogn.hpp"
#include "Preprocessing.hpp"
#include "Segmentation.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <iostream>

int main(int argc, char *argv[]) {
    if (argc != 2) {
        std::cout << "usage: " << argv[0] << " <filename>" << std::endl;
    }
    cv::Mat image = cv::imread(argv[1]);
    cv::Mat converted = preprocess(image);
    Segments s = segmentation(converted);
    cv::Mat colored(image.rows, image.cols, CV_8UC3);
    cv::cvtColor(converted, colored, CV_GRAY2RGB);
    for (size_t i = 0; i < s.size(); ++i) {
        cv::rectangle(
            colored,
            s[i],
            cv::Scalar(0, 0, 255)
        );
    }

    cv::imshow("Converted", colored);
    cv::waitKey(-1);
    return 0;
}
