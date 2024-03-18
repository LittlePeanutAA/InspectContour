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

class errorPoint
{
public:
    cv::Point2f point;
    double distance;
};

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
std::pair< std::vector<cv::Point>, std::vector<int> >
trainTemplate(const cv::Mat& temp_img, double threshold) {
    std::vector<cv::Point> temp_contour = extract_contour(temp_img, threshold);
    int height = temp_img.rows;
    int width = temp_img.cols;

    return { temp_contour, {width, height} };
}

// Convert coordinate for error point:
cv::Point convert_coor(cv::Point point, cv::Mat convert_mat) {
    int x, y, x_, y_;
    x = point.x;
    y = point.y;
    // Formula:
    x_ = round( x * convert_mat.at<double>(0, 0) + y * convert_mat.at<double>(0, 1) + convert_mat.at<double>(0, 2) );
    y_ = round( x * convert_mat.at<double>(1, 0) + y * convert_mat.at<double>(1, 1) + convert_mat.at<double>(1, 2) );
    return cv::Point(x_, y_);
}


// Define compareContour function:
std::pair<std::vector<cv::Point2f>, std::vector<cv::Point2f>>
compareContour(const std::vector<cv::Point>& temp_contour, std::vector<int> temp_size, const cv::Mat& targ_img, std::vector<cv::Point2i> rect, double threshold, int distance_thresh) {
    auto start = std::chrono::high_resolution_clock::now();
    int width = temp_size[0] - 1, height = temp_size[1] - 1;
    std::vector<cv::Point2f> src = { cv::Point2f(0, 0), cv::Point2f(width, 0), cv::Point2f(0, height) };
    std::vector<cv::Point2f> dst = { cv::Point2f(rect[0]), cv::Point2f(rect[1]), cv::Point2f(rect[3]) };
    cv::Mat convert_mat = cv::getAffineTransform(src, dst);

    // Convert coor of template contour:
    std::vector<cv::Point> converted_temp_contour;
    for (const cv::Point& point : temp_contour) {
        converted_temp_contour.push_back(cv::Point2f(convert_coor(point, convert_mat)));
    }

    // Divide bin:
    int patch_size = 2 * distance_thresh;
    std::vector<cv::Point> targ_contour = extract_contour(targ_img, threshold);
    std::map < std::pair<int, int>, std::vector<int> > temp_bin = devide_bin(converted_temp_contour, patch_size),
        targ_bin = devide_bin(targ_contour, patch_size);

    //Calculate distance and give defect position
    std::vector<cv::Point2f> targ_output, temp_output;

    // Calculate distance from target contour to template contour
    for (const cv::Point& point : targ_contour) {
        int x = point.x / patch_size, y = point.y / patch_size;   // Lẩy chỉ số của bin chứa điểm đang xét
        int k1 = (point.x % patch_size >= distance_thresh) ? 1 : -1, k2 = (point.y % patch_size >= distance_thresh) ? 1 : -1; // Lấy chỉ số của 3 bin lân cận cần xét

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
                double distance = std::abs(converted_temp_contour[idx].x - point.x) + std::abs(converted_temp_contour[idx].y - point.y);
                min_distance = std::min(min_distance, distance);
            }
            if (min_distance > distance_thresh) {
                targ_output.push_back(cv::Point2f(point));
            }
        }
        else {        // Nếu list rỗng, ta đưa mức khoảng cách về ngưỡng lỗi
            targ_output.push_back(cv::Point2f(point));
        }
    }

    // Calculate distance from coverted template contour to target contour
    for (const cv::Point& point : converted_temp_contour) {
        int x = point.x / patch_size, y = point.y / patch_size;
        int k1 = (point.x % patch_size >= distance_thresh) ? 1 : -1, k2 = (point.y % patch_size >= distance_thresh) ? 1 : -1;

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
            if (min_distance > distance_thresh) {
                targ_output.push_back(cv::Point2f(point));
            }
        }
        else {  
            targ_output.push_back(cv::Point2f(point));
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << elapsed.count() << std::endl;

    return { targ_output, temp_output };
}


/*
std::pair<std::vector<errorPoint>, std::vector<errorPoint>> 
compareContour(const std::vector<cv::Point>& temp_contour, std::vector<int> temp_size, const cv::Mat& targ_img, std::vector<cv::Point2i> rect, double threshold, int distance_thresh) {
    auto start = std::chrono::high_resolution_clock::now();
    int width = temp_size[0] - 1, height = temp_size[1] - 1;
    std::vector<cv::Point2f> src = { cv::Point2f(0, 0), cv::Point2f(width, 0), cv::Point2f(0, height) };
    std::vector<cv::Point2f> dst = { cv::Point2f(rect[0]), cv::Point2f(rect[1]), cv::Point2f(rect[3]) };
    cv::Mat convert_mat = cv::getAffineTransform(src, dst);

    // Convert coor of template contour:
    std::vector<cv::Point> converted_temp_contour;
    for (const cv::Point& point : temp_contour) {
        converted_temp_contour.push_back(cv::Point2f(convert_coor(point, convert_mat)));
    }

    // Divide bin:
    int patch_size = 2 * distance_thresh;
    std::vector<cv::Point> targ_contour = extract_contour(targ_img, threshold);
    std::map < std::pair<int, int>, std::vector<int> > temp_bin = devide_bin(converted_temp_contour, patch_size),
        targ_bin = devide_bin(targ_contour, patch_size);
   
    //Calculate distance and give defect position
    std::vector<errorPoint> targ_output, temp_output;

    // Calculate distance from target contour to template contour
    for (const cv::Point& point : targ_contour) {
        int x = point.x / patch_size, y = point.y / patch_size;   // Lẩy chỉ số của bin chứa điểm đang xét
        int k1 = (point.x % patch_size >= distance_thresh) ? 1 : -1, k2 = (point.y % patch_size >= distance_thresh) ? 1 : -1; // Lấy chỉ số của 3 bin lân cận cần xét

        std::vector<std::pair<int, int>> key_vec{ {x, y}, {x + k1, y}, {x, y + k2}, {x + k1, y + k2} };
        // Tạo list gồm các index trong các bin
        std::vector<int> idx_list;
        for (const auto& k : key_vec) {
            if (temp_bin.count(k) > 0) {
                idx_list.insert(idx_list.end(), temp_bin[k].begin(), temp_bin[k].end());
            }
        }

        errorPoint ER;
        if (idx_list.size() != 0) {     // Nếu list không rỗng, ta tính toán với các điểm trong list
            double min_distance = std::numeric_limits<double>::max();
            
            for (const auto& idx : idx_list) {
                double distance = std::abs(converted_temp_contour[idx].x - point.x) + std::abs(converted_temp_contour[idx].y - point.y);
                min_distance = std::min(min_distance, distance);
            }
            if (min_distance > distance_thresh) {
                ER.point = point;
                ER.distance = min_distance;
                targ_output.push_back( ER );
            }
        }
        else {        // Nếu list rỗng, ta đưa mức khoảng cách về ngưỡng lỗi
            ER.point = point;
            ER.distance = distance_thresh + 1;
            targ_output.push_back( ER );
        }
    }

    // Calculate distance from coverted template contour to target contour
    for (const cv::Point& point : converted_temp_contour) {
        int x = point.x / patch_size, y = point.y / patch_size;
        int k1 = (point.x % patch_size >= distance_thresh) ? 1 : -1, k2 = (point.y % patch_size >= distance_thresh) ? 1 : -1;

        std::vector<std::pair<int, int>> key_vec{ {x, y}, {x + k1, y}, {x, y + k2}, {x + k1, y + k2} };
        std::vector<int> idx_list;
        for (const auto& k : key_vec) {
            if (targ_bin.count(k) > 0) {
                idx_list.insert(idx_list.end(), targ_bin[k].begin(), targ_bin[k].end());
            }
        }

        errorPoint ER;
        if (idx_list.size() != 0) {
            double min_distance = std::numeric_limits<double>::max();
            for (const auto& idx : idx_list) {
                double distance = std::abs(targ_contour[idx].x - point.x) + std::abs(targ_contour[idx].y - point.y);
                min_distance = std::min(min_distance, distance);
            }
            if (min_distance > distance_thresh) {
                ER.point = point;
                ER.distance = min_distance;
                targ_output.push_back(ER);
            }
        }
        else {        // Nếu list rỗng, ta đưa mức khoảng cách về ngưỡng lỗi
            ER.point = point;
            ER.distance = distance_thresh + 1;
            targ_output.push_back(ER);
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << elapsed.count() << std::endl;

    return { targ_output, temp_output };    
}*/