#ifndef CONTOURRECOGNITION_HPP
#define CONTOURRECOGNITION_HPP

#include <vector>

#include "opencv2/core/core.hpp"

/*
 * A single contour, represented by its bounding box and length of the perimeter
 * in pixels
 */
typedef std::pair<cv::Rect, unsigned> Contour;

/*
 * Collection of contours
 */
typedef std::vector<Contour> Contours;

/*
 * Naïve implementation of square-based contour recognition for grayscale images.
 * size_percentage_threshold allows for filtering of the small contours, which
 * area is lesser than the given percentage of the whole image.
 */
Contours contour_recognition(const cv::Mat& mat, double size_percentage_threshold = 0.001);

#endif /* ifndef CONTOURRECOGNITION_HPP */
