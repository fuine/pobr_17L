#ifndef FEATURES_HPP
#define FEATURES_HPP

#include <vector>

#include "opencv2/core/core.hpp"

#include "Segmentation.hpp"

typedef std::vector<std::vector<double>> Moments;
typedef std::vector<double> Invariants;
typedef std::vector<double> Coefficients;

class Features {
    public:
        /*
         * ID of the image
         */
        unsigned image_id;
        /*
         * ID of the segment
         */
        unsigned segment_id;
        /*
         * Normal moments M_{p,q}, where p and q in [0,3]
         */
        Moments normal_moments;
        /*
         * Central moments M_{p,q}, where p and q in [0,3]
         */
        Moments central_moments;
        /*
         * Moment invariants from 1 to 10
         */
        Invariants invariants;
        /*
         * Various coefficients:
         * 0, 1: circularity coefficients
         * 2: Malinowska's coefficient
         * 3: Mz coefficient
         */
        Coefficients coeffs;

        /*
         * C-tor for Features, calculates all possible features
         */
        Features(const cv::Mat& segment, unsigned perimeter, unsigned image_id, unsigned segment_id);

        /*
         * Create csv representation of the calculated features
         */
        std::string as_csv_row() const;
        /*
         * Create header for the csv representation of the features
         */
        std::string get_csv_header() const;

    private:
        /*
         * Default c-tor disallowed
         */
        Features() = delete;
        /*
         * Calculate normal moments for the given segment
         */
        void calc_normal_moments(const cv::Mat& segment);
        /*
         * Calculate central moments based on the normal moments for the segment
         */
        void calc_central_moments();
        /*
         * Calculate moment invariants based on the normal and central moments of the segment
         */
        void calc_moment_invariants();
        /*
         * Calculate various coefficients for the segment
         */
        void calc_coefficients(unsigned S, unsigned L);
        /*
         * Calculate area of the segment
         */
        unsigned get_area(const cv::Mat& mat) const;
};

/*
 * Get features for each segment in the collection
 */
std::vector<Features> get_features_for_segments(const cv::Mat& image, unsigned image_id, const Segments& segments);

#endif /* ifndef FEATURES_HPP */
