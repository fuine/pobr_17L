#ifndef SEGMENTATION_HPP
#define SEGMENTATION_HPP

#include "opencv2/core/core.hpp"
#include <vector>

typedef std::vector<cv::Rect> Segments;

/*
 * Naïve implementation of square-based segmentation for grayscale images.
 * size_percentage_threshold allows for filtering of the small segments, which
 * area is lesser than the given percentage of the whole image.
 */
Segments segmentation(const cv::Mat& mat, double size_percentage_threshold = 0.001);

#endif /* ifndef SEGMENTATION_HPP */
