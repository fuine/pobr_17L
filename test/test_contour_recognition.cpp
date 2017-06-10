#define BOOST_TEST_MODULE ContourRecognitionTest

#include <boost/test/unit_test.hpp>
#include <boost/utility/binary.hpp>
#include <opencv2/core/core.hpp>
#include "opencv2/highgui/highgui.hpp"
#include <iostream>
#include <memory>

#include "ContourRecognition.hpp"


BOOST_AUTO_TEST_SUITE(test_contour_recognition)

BOOST_AUTO_TEST_CASE( TestContourRecognitionOnePixel ) {
   cv::Mat m = cv::imread("./test_files/contour_recognition_1.png", cv::IMREAD_GRAYSCALE);
   std::cout << m.rows << " " << m.cols << std::endl;
   Contours s = contour_recognition(m, 0.);
   BOOST_CHECK_EQUAL(s.size(), 1);
}

BOOST_AUTO_TEST_SUITE_END()
