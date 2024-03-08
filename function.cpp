#include <opencv2/opencv.hpp>
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/core/core.hpp"
#include <vector>
#include <map>
#include <cmath>
#include <algorithm>
#include <sys/stat.h>


// Define function - extract contour and divide points into bins
std::pair<std::vector<cv::Point>, std::map<std::pair<int, int>, std::vector<int>>>
process_contour(const cv::Mat& image, double threshold, int stride) {

    // Threshold image input and save into bin_img
    cv::Mat bin_img;
    cv::threshold(image, bin_img, threshold, 255, cv::THRESH_BINARY_INV);
    // Use function findContours in OpenCV to extract contours of bin_img
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(bin_img, contours, cv::RETR_TREE, cv::CHAIN_APPROX_NONE);

    // Remove contours having length smaller than 100
    contours.erase(std::remove_if(contours.begin(), contours.end(), [](const std::vector<cv::Point>& cnt) {
        return cv::arcLength(cnt, true) <= 100;
        }), contours.end());

    // Concatenate contours (a vector of vectors containing points) into contour (a vector containing points)
    std::vector<cv::Point> contour;
    for (const auto& cnt : contours) {
        contour.insert(contour.end(), cnt.begin(), cnt.end());
    }

    // Divide points in contour into bins
    std::map<std::pair<int, int>, std::vector<int>> bin;
    for (int idx = 0; idx < contour.size(); ++idx) {
        auto& point = contour[idx];
        bin[std::make_pair(point.x / stride, point.y / stride)].push_back(idx);
    }
    return { contour, bin };
}



// Define trainTemplate function:
std::pair<std::vector<cv::Point>, std::map<std::pair<int, int>, std::vector<int>>>
trainTemplate(const cv::Mat& temp_img, double threshold, int distance_thresh) {
    int patch_size = 2 * distance_thresh;
    std::vector<cv::Point> temp_contour;
    std::map < std::pair<int, int>, std::vector<int> > temp_bin;
    tie(temp_contour, temp_bin) = process_contour(temp_img, threshold, patch_size);
    return { temp_contour, temp_bin };
}


// Crop image from 4 corner points
cv::Mat
four_point_transform(cv::Mat image, std::vector<cv::Point2d> rect) {
    cv::Point2f tl = cv::Point2f(rect[0]);
    cv::Point2f tr = cv::Point2f(rect[1]);
    cv::Point2f br = cv::Point2f(rect[2]);
    cv::Point2f bl = cv::Point2f(rect[3]);

    float widthA = std::sqrt(std::pow(br.x - bl.x, 2) + std::pow(br.y - bl.y, 2));
    float widthB = std::sqrt(std::pow(tr.x - tl.x, 2) + std::pow(tr.y - tl.y, 2));
    int maxWidth = std::max(static_cast<int>(widthA), static_cast<int>(widthB));

    float heightA = std::sqrt(std::pow(tr.x - br.x, 2) + std::pow(tr.y - br.y, 2));
    float heightB = std::sqrt(std::pow(tl.x - bl.x, 2) + std::pow(tl.y - bl.y, 2));
    int maxHeight = std::max(static_cast<int>(heightA), static_cast<int>(heightB));

    //cv::Mat dst = (cv::Mat_<int>(4, 2) << 0, 0, maxWidth - 1, 0, maxWidth - 1, maxHeight - 1, 0, maxHeight - 1);
    std::vector<cv::Point2f> dst = { cv::Point2f(0, 0), cv::Point2f(maxWidth - 1, 0), cv::Point2f(maxWidth - 1, maxHeight - 1), cv::Point2f(0, maxHeight - 1) };
    std::vector<cv::Point2f> new_rect = { tl, tr, br, bl };

    cv::Mat M = cv::getPerspectiveTransform(new_rect, dst);
    cv::Mat warped;
    cv::warpPerspective(image, warped, M, cv::Size(maxWidth, maxHeight));

    return warped;
}

// Define compareContour function:
std::pair<std::vector<cv::Point2f>, std::vector<cv::Point2f>>
compareContour(const std::vector<cv::Point>& temp_contour, std::map<std::pair<int, int>, std::vector<int>> temp_bin, const cv::Mat& targ_img, std::vector<cv::Point2d> rect, double threshold, int distance_thresh) {
    auto start = std::chrono::high_resolution_clock::now();

    int patch_size = 2 * distance_thresh;
    cv::Mat cropped_targ_img;
    cropped_targ_img = four_point_transform(targ_img, rect);

    // Find contour and divide points into bins
    std::vector<cv::Point> targ_contour;
    std::map<std::pair<int, int>, std::vector<int>> targ_bin;
    tie(targ_contour, targ_bin) = process_contour(cropped_targ_img, threshold, patch_size);

    // Calculate distance 1
    std::vector<double> distance_1(targ_contour.size());
    for (int i = 0; i < targ_contour.size(); i++) {
        const cv::Point& point = targ_contour[i];   // Chạy lần lượt các điểm thuộc targ_contour
        int x = point.x / patch_size;
        int y = point.y / patch_size;   // Lẩy chỉ số của bin chứa điểm đang xét
        // Lấy chỉ số của 3 bin lân cận cần xét
        int k1 = (point.x % patch_size >= distance_thresh) ? 1 : -1;
        int k2 = (point.y % patch_size >= distance_thresh) ? 1 : -1;
        std::vector<std::pair<int, int>> key_vec{ {x, y}, {x + k1, y}, {x, y + k2}, {x + k1, y + k2} };
        // Tạo list gồm các index trong các bin
        std::vector<int> idx_list;
        for (const auto& k : key_vec) {
            if (temp_bin.count(k) > 0) {
                idx_list.insert(idx_list.end(), temp_bin[k].begin(), temp_bin[k].end());
            }
        }
        if (idx_list.size() != 0) {     // Nếu list không rỗng, ta tính toán với các điểm trong list
            double min_distance = std::numeric_limits<double>::max();
            for (const auto& idx : idx_list) {
                double distance = std::abs(temp_contour[idx].x - point.x) + std::abs(temp_contour[idx].y - point.y);
                min_distance = std::min(min_distance, distance);
            }
            distance_1[i] = min_distance;
        }
        else {        // Nếu list rỗng, ta đưa mức khoảng cách về ngưỡng lỗi
            distance_1[i] = distance_thresh + 1;
        }
    }

    // Calculate distance 2
    std::vector<double> distance_2(temp_contour.size());
    for (int i = 0; i < temp_contour.size(); i++) {
        const cv::Point& point = temp_contour[i];
        int x = point.x / patch_size;
        int y = point.y / patch_size;
        int k1 = (point.x % patch_size >= distance_thresh) ? 1 : -1;
        int k2 = (point.y % patch_size >= distance_thresh) ? 1 : -1;
        std::vector<std::pair<int, int>> key_vec{ {x, y}, {x + k1, y}, {x, y + k2}, {x + k1, y + k2} };
        std::vector<int> idx_list;
        for (const auto& k : key_vec) {
            if (targ_bin.count(k) > 0) {
                idx_list.insert(idx_list.end(), targ_bin[k].begin(), targ_bin[k].end());
            }
        }
        if (idx_list.size() != 0) {
            double min_distance = std::numeric_limits<double>::max();
            for (const auto& idx : idx_list) {
                double distance = std::abs(targ_contour[idx].x - point.x) + std::abs(targ_contour[idx].y - point.y);
                min_distance = std::min(min_distance, distance);
            }
            distance_2[i] = min_distance;
        }
        else {
            distance_2[i] = distance_thresh + 1;
        }
    }

    // Calculate position distance greater than distance_thresh
    std::vector<cv::Point2f> pos_1, pos_2;
    for (int i = 0; i < distance_1.size(); i++) {
        if (distance_1[i] > distance_thresh) {
            pos_1.push_back(cv::Point2f(targ_contour[i]));
        }
    }
    for (int i = 0; i < distance_2.size(); i++) {
        if (distance_2[i] > distance_thresh) {
            pos_2.push_back(cv::Point2f(temp_contour[i]));
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << elapsed.count() << std::endl;

    return { pos_1, pos_2 };
}