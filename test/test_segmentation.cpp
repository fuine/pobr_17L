#define BOOST_TEST_MODULE SegmentationTest

#include <boost/test/unit_test.hpp>
#include <boost/utility/binary.hpp>
#include <opencv2/core/core.hpp>
#include "opencv2/highgui/highgui.hpp"
#include <iostream>
#include <memory>

#include "Segmentation.hpp"


BOOST_AUTO_TEST_SUITE(test_segmentation)

BOOST_AUTO_TEST_CASE( TestSegmentationOnePixel ) {
   cv::Mat m = cv::imread("./test_files/segmentation_test_1.png", cv::IMREAD_GRAYSCALE);
   std::cout << m.rows << " " << m.cols << std::endl;
   Segments s = segmentation(m, 0.);
   BOOST_CHECK_EQUAL(s.size(), 1);
}

BOOST_AUTO_TEST_SUITE_END()
