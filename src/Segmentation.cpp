#include "Segmentation.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>
#include <cmath>

// channel weights used in the grayscale transform
static int BLUE_WEIGHT = 2;
static int GREEN_WEIGHT = -2;
static int RED_WEIGHT = -2;

/*
 * Equalize colors' histograms for the BGR image.
 * https://en.wikipedia.org/wiki/Histogram_equalization
 */
void histogram_equalization(cv::Mat& m) {
    // calculate histograms
    double histograms[3][256] = {{0.}, {0.}, {0.}};
    for(int i = 0; i < m.rows; ++i) {
        cv::Vec3b* m_i = m.ptr<cv::Vec3b>(i);
        for(int j = 0; j < m.cols; ++j) {
            cv::Vec3b& m_ij = m_i[j];
            ++histograms[0][m_ij[0]];
            ++histograms[1][m_ij[1]];
            ++histograms[2][m_ij[2]];
        }
    }

    // normalize histograms
    double total = static_cast<double>(m.rows * m.cols);
    histograms[0][0] /= total;
    histograms[1][0] /= total;
    histograms[2][0] /= total;
    for(int i = 0; i < 3; ++i) {
        double prev = histograms[i][0];
        for(int j = 1; j < 256; ++j) {
            prev = (histograms[i][j] / total) + prev;
            histograms[i][j] = prev;
        }
    }

    // modify the original image
    for(int i = 0; i < m.rows; ++i) {
        cv::Vec3b* m_i = m.ptr<cv::Vec3b>(i);
        for(int j = 0; j < m.cols; ++j) {
            cv::Vec3b& m_ij = m_i[j];
            m_ij[0] = static_cast<unsigned char>(floor(255 * histograms[0][m_ij[0]]));
            m_ij[1] = static_cast<unsigned char>(floor(255 * histograms[1][m_ij[1]]));
            m_ij[2] = static_cast<unsigned char>(floor(255 * histograms[2][m_ij[2]]));
        }
    }
}

/*
 * Enhance color values by stretching them in the HSV space.
 * See https://git.gnome.org/browse/gimp/tree/plug-ins/common/color-enhance.c for more informations.
 */
void color_enhance(cv::Mat& mat) {
    cv::Mat tmp_mat = mat;
    unsigned c;
    unsigned y;
    unsigned m;
    unsigned k;
    std::vector<unsigned> ks(mat.rows*mat.cols);

    // convert to HSV
    for(int i = 0; i < tmp_mat.rows; ++i) {
        cv::Vec3b* m_i = tmp_mat.ptr<cv::Vec3b>(i);
        for(int j = 0; j < tmp_mat.cols; ++j) {
            cv::Vec3b& m_ij = m_i[j];
            c = 255 - m_ij[0];
            m = 255 - m_ij[1];
            y = 255 - m_ij[2];

            k = c;
            if (m < k) {
                k = m;
            }
            if (y < k) {
                k = y;
            }
            m_ij[0] = static_cast<unsigned char>(c - k);
            m_ij[1] = static_cast<unsigned char>(m - k);
            m_ij[2] = static_cast<unsigned char>(y - k);
            ks[mat.cols*i + j] = k;
        }
    }

    cv::cvtColor(tmp_mat, tmp_mat, CV_BGR2HSV);
    unsigned vhi = 0;
    unsigned vlo = 255;

    // find vhi and vlo
    for(int i = 0; i < tmp_mat.rows; ++i) {
        cv::Vec3b* m_i = tmp_mat.ptr<cv::Vec3b>(i);
        for(int j = 0; j < tmp_mat.cols; ++j) {
            cv::Vec3b& m_ij = m_i[j];
            if (m_ij[2] > vhi) {
                vhi = m_ij[2];
            }
            if (m_ij[2] < vlo) {
                vlo = m_ij[2];
            }
        }
    }

    // enhance image
    if (vhi == vlo) {
        return;
    }

    for(int i = 0; i < tmp_mat.rows; ++i) {
        cv::Vec3b* m_i = tmp_mat.ptr<cv::Vec3b>(i);
        for(int j = 0; j < tmp_mat.cols; ++j) {
            cv::Vec3b& m_ij = m_i[j];
            // this will never overflow/underflow
            m_ij[2] = static_cast<unsigned char>(((m_ij[2] - vlo) * 255) / (vhi - vlo));
        }
    }

    // Convert back to BGR
    cv::cvtColor(tmp_mat, tmp_mat, CV_HSV2BGR);
    for(int i = 0; i < tmp_mat.rows; ++i) {
        cv::Vec3b* m_i = tmp_mat.ptr<cv::Vec3b>(i);
        for(int j = 0; j < tmp_mat.cols; ++j) {
            cv::Vec3b& m_ij = m_i[j];
            c = m_ij[0];
            m = m_ij[1];
            y = m_ij[2];
            k = ks[mat.cols*i + j];

            c += k;
            if (c > 255) {
                c = 255;
            }
            m += k;
            if (m > 255) {
                m = 255;
            }
            y += k;
            if (y > 255) {
                y = 255;
            }

            m_ij[0] = static_cast<unsigned char>(255 - c);
            m_ij[1] = static_cast<unsigned char>(255 - m);
            m_ij[2] = static_cast<unsigned char>(255 - y);
        }
    }
    mat = tmp_mat;
}

/*
 * Transform colored image in BGR to grayscale, using specified weights
 */
cv::Mat color_to_grayscale(const cv::Mat& m, int blue_weight, int green_weight, int red_weight) {
    cv::Mat grayscale(m.rows, m.cols, CV_8UC1);
    int gray_pixel;
    for(int i = 0; i < m.rows; ++i) {
        const cv::Vec3b* m_i = m.ptr<cv::Vec3b>(i);
        unsigned char* g_i = grayscale.ptr<unsigned char>(i);
        for(int j = 0; j < m.cols; ++j) {
            const cv::Vec3b& m_ij = m_i[j];
            gray_pixel = (m_ij[0] * blue_weight) + (m_ij[1] * green_weight) + (m_ij[2] * red_weight);
            if (gray_pixel >= 256) {
                gray_pixel = 255;
            }
            else if (gray_pixel < 0) {
                gray_pixel = 0;
            }
            g_i[j] = static_cast<unsigned char>(gray_pixel);
        }
    }
    return grayscale;
}

cv::Mat preprocess(const cv::Mat& m) {
    cv::Mat _I = m;
    histogram_equalization(_I);
    color_enhance(_I);
    cv::Mat grayscale = color_to_grayscale(_I, BLUE_WEIGHT, GREEN_WEIGHT, RED_WEIGHT);
    return grayscale;
}
