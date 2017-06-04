#ifndef MOMENTS_HPP
#define MOMENTS_HPP

#include "opencv2/core/core.hpp"
#include <vector>

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

        Features(const cv::Mat& segment, unsigned image_id, unsigned segment_id);

        std::string as_csv_row() const;
        std::string get_csv_header() const;
    private:
        Features() = delete;
        void calc_normal_moments(const cv::Mat& segment);
        void calc_central_moments();
        void calc_moment_invariants();
};

std::vector<Features> get_features_for_segments(const cv::Mat& image, unsigned image_id, const std::vector<cv::Rect>& segments);

#endif /* ifndef MOMENTS_HPP */
