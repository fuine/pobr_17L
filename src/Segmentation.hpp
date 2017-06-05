#ifndef SEGMENTATION_HPP
#define SEGMENTATION_HPP

#include <vector>

#include "opencv2/core/core.hpp"

/*
 * A single segment, represented by its bounding box and length of the perimeter
 * in pixels
 */
typedef std::pair<cv::Rect, unsigned> Segment;

/*
 * Collection of segments
 */
typedef std::vector<Segment> Segments;

/*
 * Naïve implementation of square-based segmentation for grayscale images.
 * size_percentage_threshold allows for filtering of the small segments, which
 * area is lesser than the given percentage of the whole image.
 */
Segments segmentation(const cv::Mat& mat, double size_percentage_threshold = 0.001);

#endif /* ifndef SEGMENTATION_HPP */
