#include <cmath>

#include "opencv2/imgproc.hpp"

#include "Preprocessing.hpp"

// channel weights used in the grayscale transform
static int BLUE_WEIGHT = 2;
static int GREEN_WEIGHT = -2;
static int RED_WEIGHT = -2;

// size of the spatial filters
static int SPATIAL_FILTER_SIZE = 3;

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

/*
 * Create pixel by sum of element-wise multiplication of the matrix and filter.
 */
unsigned char multiply(const cv::Mat& mat, const cv::Mat& filter) {
    int sum = 0;
    int filter_size = filter.rows;
    for (int i = 0; i < filter_size; ++i) {
        for (int j = 0; j < filter_size; ++j){
            sum += static_cast<int>(mat.at<uchar>(i, j) * filter.at<float>(i, j));
        }
    }
    if (sum >= 256){
        return 255;
    }
    else if (sum <= 0) {
        return 0;
    }
    else {
        return static_cast<unsigned char>(sum);
    }
}

/*
 * Apply spatial filter to the image.
 */
void apply_spatial_filter(cv::Mat& mat, const cv::Mat& filter) {
    int filter_size = filter.rows;
    cv::Mat tmp(filter_size, filter_size, CV_8UC3);

    for (int i = 0; i < mat.rows - (filter_size + 1); ++i) {
        unsigned char* g_i = mat.ptr<unsigned char>(i);
        for (int j = 0; j < mat.cols - (filter_size + 1); ++j) {
            cv::Mat tmp = mat(cv::Rect(j, i, filter_size, filter_size));
            g_i[j] = multiply(tmp, filter);
        }
    }
}

/*
 * Make pixels below given threshold black.
 */
void threshold(cv::Mat& mat, int threshold) {
    for (int i = 0; i < mat.rows; ++i) {
        unsigned char* g_i = mat.ptr<unsigned char>(i);
        for (int j = 0; j < mat.cols; ++j) {
            if (g_i[j] <= threshold) {
                g_i[j] = 0;
            }
        }
    }
}

/*
 * Convert BGR image to its thresholded grayscale representation
 */
cv::Mat preprocess(const cv::Mat& mat) {
    cv::Mat mask1(SPATIAL_FILTER_SIZE, SPATIAL_FILTER_SIZE, CV_32FC1, 0.2);
    cv::Mat mask2(SPATIAL_FILTER_SIZE, SPATIAL_FILTER_SIZE, CV_32FC1, -1);
    int center = SPATIAL_FILTER_SIZE / 2;
    mask2.at<float>(center, center) = static_cast<float>((SPATIAL_FILTER_SIZE * SPATIAL_FILTER_SIZE) + 1);

    cv::Mat image = mat.clone();
    histogram_equalization(image);
    color_enhance(image);
    cv::Mat grayscale = color_to_grayscale(image, BLUE_WEIGHT, GREEN_WEIGHT, RED_WEIGHT);
    apply_spatial_filter(grayscale, mask1);
    threshold(grayscale, 50);
    apply_spatial_filter(grayscale, mask2);
    threshold(grayscale, 150);
    return grayscale;
}
