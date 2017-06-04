#include "ImageRecogn.hpp"
#include "Preprocessing.hpp"
#include "Segmentation.hpp"
#include "Moments.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <iostream>
#include <sstream>
#include <iostream>
#include <fstream>

int main(int argc, char *argv[]) {
    if (argc != 2) {
        std::cout << "usage: " << argv[0] << " <filename>" << std::endl;
    }
    cv::Mat image = cv::imread(argv[1]);
    if (image.rows == 0 || image.cols == 0) {
        std::cout << "Something went wrong while trying to read image " << argv[1] << std::endl;
        return -1;
    }
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
        char t[3];
        snprintf(t, 3, "%d", static_cast<int>(i));
        int x = s[i].x + 7;
        int y = s[i].y + s[i].height - 7;
        cv::putText(colored, t, cv::Point(x, y), cv::FONT_HERSHEY_PLAIN, 2, cv::Scalar(0, 0, 255), 2);
    }
    cv::putText(colored, argv[1], cv::Point(7, image.rows - 7), cv::FONT_HERSHEY_PLAIN, 2, cv::Scalar(255, 0, 0), 2);

    unsigned image_id = static_cast<unsigned>(round(1000000 * rand()));
    if (s.size() > 0) {
        std::vector<Features> fs = get_features_for_segments(converted, image_id, s);
        std::ofstream myfile;
        std::stringstream ss;
        ss << argv[1] << "_features.csv";
        std::cout << "Saving to: " << ss.str() << std::endl;
        myfile.open(ss.str());
        myfile << fs[0].get_csv_header() << std::endl;
        for (auto i : fs) {
            myfile << i.as_csv_row() << std::endl;
        }
        myfile.close();
    }

    std::cout << "Limit: " << std::numeric_limits<double>::max() << std::endl;
    cv::imshow("Converted", colored);
    cv::waitKey(-1);
    return 0;
}
