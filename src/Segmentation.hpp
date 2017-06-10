#ifndef SEGMENTATION_HPP
#define SEGMENTATION_HPP

#include "opencv2/core/core.hpp"

/*
 * Convert BGR image to its thresholded grayscale representation
 */
cv::Mat segmentation(const cv::Mat& mat);

#endif /* ifndef SEGMENTATION_HPP */
