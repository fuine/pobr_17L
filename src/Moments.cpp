#include "Moments.hpp"
#include <sstream>

void Features::calc_normal_moments(const cv::Mat& mat) {
    // normal moments
    // Moments m = {{0., 0., 0.},{0., 0., 0.},{0., 0., 0.}};
    Moments m(4, std::vector<double>(4, 0.));
    for (int i = 0; i < mat.rows; ++i) {
        const unsigned char* g_i = mat.ptr<unsigned char>(i);
        for (int j = 0; j < mat.cols; ++j) {
            if (g_i[j] == 0) {
                for (int p = 0; p < 4; ++p) {
                    for (int q = 0; q < 4; ++q) {
                        m[p][q] += pow(i, p) * pow(j, q);
                    }
                }
            }
        }
    }
    this->normal_moments = m;
}



void Features::calc_central_moments() {
    Moments M(4, std::vector<double>(4, 0.));
    Moments m = this->normal_moments;
    double i_c = m[1][0] / m[0][0];
    double j_c = m[0][1] / m[0][0];

    M[0][0] = m[0][0];
    M[1][1] = m[1][1] - m[1][0]*m[0][1] / m[0][0];

    M[2][0] = m[2][0] - pow(m[1][0], 2) / m[0][0];
    M[0][2] = m[0][2] - pow(m[0][1], 2) / m[0][0];
    M[2][1] = m[2][1] - 2.0*m[1][1]*i_c - m[2][0]*j_c + 2.0*m[0][1]*pow(i_c, 2);
    M[1][2] = m[1][2] - 2.0*m[1][1]*j_c - m[0][2]*i_c + 2.0*m[1][0]*pow(j_c, 2);

    M[0][3] = m[0][3] - 3.0*m[0][2]*j_c + 2.0*m[0][1] *pow(j_c, 2);
    M[3][0] = m[3][0] - 3.0*m[2][0]*i_c + 2.0*m[1][0]*pow(i_c, 2);
    this->central_moments = M;
}

void Features::calc_moment_invariants() {
    Invariants I(11, 0.);
    Moments& m = this->normal_moments;
    Moments& M = this->central_moments;

    I[1] = (M[2][0] + M[0][2]) / pow(m[0][0], 2);
    I[2] = (pow(M[2][0] - M[0][2], 2) + 4.0 * pow(M[1][1], 2)) / pow(m[0][0], 4);
    I[3] = (pow(M[3][0] - 3.0 * M[1][2], 2) + pow(3.0 * M[2][1] - M[0][3], 2)) / pow(m[0][0], 5);
    I[4] = (pow(M[3][0] + M[1][2], 2) + pow(M[2][1] + M[0][3], 2)) / pow(m[0][0], 5);
    I[5] = ((M[3][0] - 3.0 * M[1][2]) * (M[3][0] + M[1][2])
            * (pow(M[3][0] + M[1][2], 2) - 3.0 * pow(M[2][1] + M[0][3], 2))
            + (3.0 * M[2][1] - M[0][3]) * (M[2][1] + M[0][3])
            * (3.0 * pow(M[3][0] + M[1][2], 2) - pow(M[2][1] + M[0][3], 2)))
           / pow(m[0][0], 10);
    I[6] = ((M[2][0] - M[0][2]) * (pow(M[3][0] + M[1][2], 2) - pow(M[2][1] + M[0][3], 2))
            + 4.0 * M[1][1] * (M[3][0] + M[1][2]) * (M[2][1] + M[0][3]))
           / pow(m[0][0], 7);
    I[7] = (M[2][0] * M[0][2] - pow(M[1][1], 2)) / pow(m[0][0], 4);
    I[8] = (M[3][0] * M[1][2] + M[2][1] * M[0][3] - pow(M[1][2], 2) - pow(M[2][1], 2)) / pow(m[0][0], 5);
    I[9] = (M[2][0] * (M[2][1] * M[0][3] - pow(M[1][2], 2))
            + M[0][2] * (M[0][3] * M[1][2] - pow(M[2][1], 2))
            - M[1][1] * (M[3][0] * M[0][3] - M[2][1] * M[1][2]))
           / pow(m[0][0], 7);
    I[10] = (pow(M[3][0] * M[0][3] - M[2][1] * M[1][2], 2)
             - 4.0 * (M[3][0] * M[1][2] - pow(M[2][1], 2)) * (M[0][3] * M[2][1] - M[1][2]))
            / pow(m[0][0], 10);
    this->invariants = I;
}

// std::pair<double, double> get_i_j(const cv::Mat& m){
    // double m01 = 0;
    // double m10 = 0;
    // double m00 = 0;

    // cv::Mat_<cv::Vec3b> _I = m;
    // for (int i = 0; i < _I.rows; ++i) {
        // for (int j = 0; j < _I.cols; ++j) {
            // if (_I(i, j)[0] == 0) {
                // m00 += 1;
                // m01 += j;
                // m10 += i;
            // }
        // }
    // }
    // return std::make_pair(m10 / m00, m01 / m00);
// }

Features::Features(const cv::Mat& segment, unsigned image_id, unsigned segment_id) {
    this->segment_id = segment_id;
    this->image_id = image_id;
    this->calc_normal_moments(segment);
    this->calc_central_moments();
    this->calc_moment_invariants();
}

std::string Features::get_csv_header() const {
    std::stringstream ss;
    ss << "id,";
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            ss << "normal" << i << j << ",";
        }
    }

    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            ss << "central" << i << j << ",";
        }
    }

    for (int i = 0; i < 11; ++i) {
        ss << "invariant" << i << ",";
    }
    ss << "is_logo,is_c";
    std::string s = ss.str();
    // remove the last comma
    // s.pop_back();
    return s;
}

std::string Features::as_csv_row() const {
    std::stringstream ss;
    ss << this->image_id << "_" << this->segment_id << ",";

    for (auto& ms : this->normal_moments) {
        for (auto& m : ms) {
            ss << m << ",";
        }
    }

    for (auto& ms : this->central_moments) {
        for (auto& m : ms) {
            ss << m << ",";
        }
    }

    for (auto i : this->invariants) {
        ss << i << ",";
    }

    ss << 0 << "," << 0;
    std::string s = ss.str();
    // remove the last comma
    // s.pop_back();
    return s;
}

std::vector<Features> get_features_for_segments(const cv::Mat& image, unsigned image_id, const std::vector<cv::Rect>& segments) {
    std::vector<Features> f;
    unsigned id = 0;
    for (auto& s : segments) {
        // get roi from segment
        cv::Mat roi(image, s);
        f.push_back(Features(roi, image_id, id++));
    }
    return f;
}
