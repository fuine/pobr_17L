#ifndef FEATURES_HPP
#define FEATURES_HPP

#include "opencv2/core/core.hpp"
#include <vector>
#include "Segmentation.hpp"

typedef std::vector<std::vector<double>> Moments;
typedef std::vector<double> Invariants;
typedef std::vector<double> Coefficients;

class Features {
    public:
        unsigned image_id;
        unsigned segment_id;
        Moments normal_moments;
        Moments central_moments;
        Invariants invariants;
        Coefficients coeffs;

        Features(const cv::Mat& segment, unsigned perimeter, unsigned image_id, unsigned segment_id);

        std::string as_csv_row() const;
        std::string get_csv_header() const;
    private:
        Features() = delete;
        void calc_normal_moments(const cv::Mat& segment);
        void calc_central_moments();
        void calc_moment_invariants();
        void calc_coefficients(unsigned S, unsigned L);
        unsigned get_area(const cv::Mat& mat) const;
};

std::vector<Features> get_features_for_segments(const cv::Mat& image, unsigned image_id, const Segments& segments);

#endif /* ifndef FEATURES_HPP */
