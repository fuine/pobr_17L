#ifndef PREPROCESSING_HPP
#define PREPROCESSING_HPP

#include "opencv2/core/core.hpp"

/*
 * Convert BGR image to its thresholded grayscale representation
 */
cv::Mat preprocess(const cv::Mat& mat);

#endif /* ifndef PREPROCESSING_HPP */
