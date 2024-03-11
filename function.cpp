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
std::vector<cv::Point>
extract_contour(const cv::Mat& image, double threshold) {
    
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

    return contour;
}

std::map<std::pair<int, int>, std::vector<int>>
devide_bin(std::vector<cv::Point> contour, int stride) {
    // Divide points in contour into bins
    std::map<std::pair<int, int>, std::vector<int>> bin;
    for (int idx = 0; idx < contour.size(); ++idx) {
        auto& point = contour[idx];
        bin[std::make_pair(point.x / stride, point.y / stride)].push_back(idx);
    }
    return bin;
}



// Define trainTemplate function:
std::pair<std::vector<cv::Point>, std::map<std::pair<int, int>, std::vector<int>>>
trainTemplate(const cv::Mat& temp_img, double threshold, int distance_thresh) {
    int patch_size = 2 * distance_thresh;
    std::vector<cv::Point> temp_contour = extract_contour(temp_img, threshold);
    std::map < std::pair<int, int>, std::vector<int> > temp_bin = devide_bin(temp_contour, patch_size);

    return { temp_contour, temp_bin };
}


// Crop image from 4 corner points
/*
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
*/


// Convert coordinate for error point:
cv::Point convert_coor(cv::Point point, cv::Mat convert_mat) {
    int x, y, x_, y_;
    x = point.x;
    y = point.y;
    // Formula:
    x_ = x * convert_mat.at<double>(0, 0) + y * convert_mat.at<double>(0, 1) + convert_mat.at<double>(0, 2);
    y_ = x * convert_mat.at<double>(1, 0) + y * convert_mat.at<double>(1, 1) + convert_mat.at<double>(1, 2);
    return cv::Point(x_, y_);
}

/*
void compareContour(const std::vector<cv::Point>& temp_contour, std::map<std::pair<int, int>, std::vector<int>> temp_bin, const cv::Mat& targ_img, std::vector<cv::Point2d> rect, double threshold, int distance_thresh) {
    auto start = std::chrono::high_resolution_clock::now();

    // Sinh ra ma trận chuyển từ các cặp điểm trong rect: convert_mat biến đổi toạ độ từ targ_img lớn sang targ_img nhỏ, inv_convert_mat ngược lại
    int height = targ_img.rows;
    int width = targ_img.cols;
    std::vector<cv::Point2f> src = { cv::Point2f(0, 0), cv::Point2f(width - 1, 0), cv::Point2f(0, height - 1) };
    std::vector<cv::Point2f> dst = { cv::Point2f(rect[0]), cv::Point2f(rect[1]), cv::Point2f(rect[3]) };
    cv::Mat convert_mat = cv::getAffineTransform(src, dst);
    cv::Mat inv_convert_mat = cv::getAffineTransform(dst, src);
    std::cout << inv_convert_mat;
}
*/


// Define compareContour function:
std::pair<std::vector<cv::Point2f>, std::vector<cv::Point2f>>
compareContour(const std::vector<cv::Point>& temp_contour, std::map<std::pair<int, int>, std::vector<int>> temp_bin, const cv::Mat& targ_img, std::vector<cv::Point2d> rect, double threshold, int distance_thresh) {
    auto start = std::chrono::high_resolution_clock::now();

    // Sinh ra ma trận chuyển từ các cặp điểm trong rect: convert_mat biến đổi toạ độ từ targ_img lớn sang targ_img nhỏ, inv_convert_mat ngược lại
    int height = targ_img.rows;
    int width = targ_img.cols;
    std::vector<cv::Point2f> src = { cv::Point2f(0, 0), cv::Point2f(width - 1, 0), cv::Point2f(0, height - 1) };
    std::vector<cv::Point2f> dst = { cv::Point2f(rect[0]), cv::Point2f(rect[1]), cv::Point2f(rect[3]) };
    cv::Mat convert_mat = cv::getAffineTransform(src, dst);
    cv::Mat inv_convert_mat = cv::getAffineTransform(dst, src);

    int patch_size = 2 * distance_thresh;

    // Find contour and divide points into bins
    std::vector<cv::Point> targ_contour = extract_contour(targ_img, threshold);
    
    std::vector<cv::Point> converted_targ_contour; 
    std::vector<cv::Point2f> pos_1, pos_2;
    double min_distance;

    // Calculate distance 1 - from targ to temp
    //std::vector<double> distance_1(targ_contour.size());
    for (int i = 0; i < targ_contour.size(); i++) {
        const cv::Point& point = targ_contour[i];   // Chạy lần lượt các điểm thuộc targ_contour
        
        cv::Point targ_pnt = convert_coor(point, convert_mat);
        converted_targ_contour.push_back(cv::Point2f(targ_pnt));
        
        int x = targ_pnt.x / patch_size;
        int y = targ_pnt.y / patch_size;   // Lẩy chỉ số của bin chứa điểm đang xét
        
        // Lấy chỉ số của 3 bin lân cận cần xét
        int k1 = (targ_pnt.x % patch_size >= distance_thresh) ? 1 : -1;
        int k2 = (targ_pnt.y % patch_size >= distance_thresh) ? 1 : -1;
        std::vector<std::pair<int, int>> key_vec{ {x, y}, {x + k1, y}, {x, y + k2}, {x + k1, y + k2} };
        
        // Tạo list gồm các index trong các bin
        std::vector<int> idx_list;
        for (const auto& k : key_vec) {
            if (temp_bin.count(k) > 0) {
                idx_list.insert(idx_list.end(), temp_bin[k].begin(), temp_bin[k].end());
            }
        }
        if (idx_list.size() != 0) {     // Nếu list không rỗng, ta tính toán với các điểm trong list
            min_distance = std::numeric_limits<double>::max();
            for (const auto& idx : idx_list) {
                double distance = std::abs(temp_contour[idx].x - targ_pnt.x) + std::abs(temp_contour[idx].y - targ_pnt.y);
                min_distance = std::min(min_distance, distance);
            }
            //distance_1[i] = min_distance;
        }
        else {        // Nếu list rỗng, ta đưa mức khoảng cách về ngưỡng lỗi
            //distance_1[i] = distance_thresh + 1;
            min_distance = distance_thresh + 1;
        }

        if (min_distance > distance_thresh) {
            pos_1.push_back(cv::Point2f(targ_contour[i]));
        }
    }
    
    // Calculate distance 2
    //std::vector<double> distance_2(temp_contour.size());
    std::map<std::pair<int, int>, std::vector<int>> targ_bin = devide_bin(converted_targ_contour, patch_size);
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
            min_distance = std::numeric_limits<double>::max();
            int min_index = 0;
            for (const auto& idx : idx_list) {
                double distance = std::abs(converted_targ_contour[idx].x - point.x) + std::abs(converted_targ_contour[idx].y - point.y);
                if (distance < min_distance) {
                    min_distance = distance;
                    min_index = idx;
                }
            }
            //distance_2[i] = min_distance;
            if (min_distance > distance_thresh) {
                pos_2.push_back(cv::Point2f(targ_contour[min_index]));
            }
        }
        else {
            //distance_2[i] = distance_thresh + 1;
            min_distance = distance_thresh + 1;
            pos_2.push_back(cv::Point2f(convert_coor(temp_contour[i], inv_convert_mat)));
        }
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << elapsed.count() << std::endl;
    
    return { pos_1, pos_2 };
    
}