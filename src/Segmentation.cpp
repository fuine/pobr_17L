#include "opencv2/core/core.hpp"

#include "Segmentation.hpp"

/*
 * Get the next clockwise pixel, starting from point b in the Moore neighbourhood
 * of the point p -- M(p)
 */
cv::Point next_clockwise_pixel(cv::Point p, cv::Point b) {
    // lookup table for the clockwise choice of the next pixel
    // first index -- 0: north 1: same level 2: south
    // second index -- 0: west 1: same level 2: east
    // third index -- 0: x offset for the new point 1: y offset
    int lut_clockwise[3][3][2] = {
        {
            // 0, 0 <-> NW -> N
            {0, -1},
            // 0, 1 <-> N -> NE
            {1, -1},
            // 0, 2 <-> NE -> E
            {1, 0},
        },
        {
            // 1, 0 <-> W -> NW
            {-1, -1},
            // 1, 1 <-> NOP
            {0, 0},
            // 1, 2 <-> E -> SE
            {1, 1},
        },
        {
            // 2, 0 <-> SW -> W
            {-1, 0},
            // 2, 1 <-> S -> SW
            {-1, 1},
            // 2, 2 <-> SE -> S
            {0, 1},
        },
    };
    // calculate differences between b and p
    int diff_rows = (b.x - p.x) + 1;
    int diff_cols = (b.y - p.y) + 1;
    // this should never happen
    assert(diff_rows != 1 || diff_cols != 1);
    // get offsets for the next clockwise pixels based on the relative position
    // of b with respect to p
    int* d = lut_clockwise[diff_cols][diff_rows];
    return cv::Point(p.x + d[0], p.y + d[1]);
}

/*
 * Modified Moore-neighbourhood tracing algorithm with Jacob Eliosoff's termination condition.
 * Used to create a rectangle on the segment via contour tracing.
 *
 * https://www.youtube.com/watch?v=UwY98217hFE
 */
Segment get_bounding_box_moore(const cv::Mat& mat, cv::Point b, cv::Point s) {
    cv::Point box_start(s.x - 1, s.y - 1);
    cv::Point box_end(s.x, s.y);
    // Set the current boundary point p to s i.e. p=s
    cv::Point p = s;
    // Let b = the pixel from which s was entered during the image scan.
    // Set c to be the next clockwise pixel (from b) in M(p).
    cv::Point c = next_clockwise_pixel(p, b);
    // Helper point, which is used to test for single-pixel segments
    cv::Point first_white = b;
    // This point is used by the Eliosoff's termination condition
    cv::Point initial_b = b;
    unsigned visited_starting_point = 0;
    unsigned perimeter_length = 0;
    while (visited_starting_point != 2) {
        // If c is white
        if (mat.at<unsigned char>(c) != 0) {
            if (visited_starting_point == 1) {
                ++perimeter_length;
            }
            // update start and endpoint of the bounding box for the segment
            if (c.x - 1 < box_start.x) {
                box_start.x = c.x - 1;
            }
            else if (c.x > box_end.x) {
                box_end.x = c.x;
            }
            if (c.y - 1 < box_start.y) {
                box_start.y = c.y - 1;
            }
            else if (c.y > box_end.y) {
                box_end.y = c.y;
            }
            b = p;
            p = c;
            // backtrack: move the current pixel c to the pixel from which p was entered
            c = next_clockwise_pixel(p, b);
            first_white = c;
        }
        else {
            // advance the current pixel c to the next clockwise pixel in M(p) and update backtrack
            b = c;
            c = next_clockwise_pixel(p, b);
            // we hit the single-pixel segment
            if (first_white == c) {
                break;
            }
        }
        if (c == s) {
            if (visited_starting_point == 0) {
                ++visited_starting_point;
                initial_b = b;
            }
            else if (b == initial_b) {
                ++visited_starting_point;
            }
        }
    }
    return std::make_pair(cv::Rect(box_start, box_end), perimeter_length);
}

/*
 * Draw a rectangle on the mat. This function is equivalent to this call:
 * cv::rectangle(mat, r, cv::Scalar(255), CV_FILLED, 8, 0);
 */
void draw_rectangle(cv::Mat& mat, const cv::Rect& r) {
    for(int i = r.y; i < r.y + r.height ; ++i) {
        unsigned char* g_i = mat.ptr<unsigned char>(i);
        g_i += r.x;
        memset(g_i, 255, r.width);
    }
}

/*
 * Naïve implementation of square-based segmentation for grayscale images.
 * size_percentage_threshold allows for filtering of the small segments, which
 * area is lesser than the given percentage of the whole image.
 */
Segments segmentation(const cv::Mat& mat, double size_percentage_threshold) {
    Segments segments;
    int minimum_size = static_cast<int>(round((mat.cols * mat.rows) * size_percentage_threshold));
    // create an artificial single-pixel black border to ensure that we always
    // have the proper backtrack point for the Moore-neighbourhood tracing
    // algorithm.
    cv::Mat bordered;
    cv::copyMakeBorder(mat, bordered, 1, 1, 1, 1, cv::BORDER_CONSTANT, cv::Scalar(0));
    // map which tracks the pixels that have already been segmented
    cv::Mat seen_map = cv::Mat::zeros(bordered.rows, bordered.cols, CV_8UC1);
    cv::Point backtrack(1, bordered.rows - 1);
    // scan the image from the south-west corner of the image, moving rows
    // first south to north, columns second west to east
    for (int j = 1; j < bordered.cols - 1; ++j) {
        for (int i = bordered.rows - 1; i >= 0; --i) {
            if (bordered.at<unsigned char>(i, j) > 0 && seen_map.at<unsigned char>(i, j) == 0) {
                Segment s = get_bounding_box_moore(bordered, backtrack, cv::Point(j, i));
                cv::Rect r = s.first;
                if (r.width * r.height >= minimum_size) {
                    segments.push_back(s);
                }
                cv::Rect seen_region(r);
                // normalize the box for the bordered image
                ++seen_region.height;
                ++seen_region.width;
                // mark the region as segmented
                draw_rectangle(seen_map, seen_region);
                i -= (r.height);
            }
            backtrack.x = j;
            backtrack.y = i;
        }
    }
    return segments;
}
