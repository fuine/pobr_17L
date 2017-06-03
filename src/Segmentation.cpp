#include "Segmentation.hpp"
#include <iostream>
#include <cmath>

// Zrównanie -> https://en.wikipedia.org/wiki/Histogram_equalization
// Uwydatnianie koloru -> https://docs.gimp.org/2.8/en/plug-in-color-enhance.html / https://git.gnome.org/browse/gimp/tree/plug-ins/common/color-enhance.c?id=bfa8547a938843102e11388bba15ad8de10877c4
// mikser kanałów -> thresholding with blue

// channel weights used in the grayscale transform
static int BLUE_WEIGHT = 2;
static int GREEN_WEIGHT = -2;
static int RED_WEIGHT = -2;

/*
 * Equalize colors' histograms for the BGR image.
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
    cv::Mat grayscale = color_to_grayscale(_I, BLUE_WEIGHT, GREEN_WEIGHT, RED_WEIGHT);
    return grayscale;
}
