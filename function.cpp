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

// Class output
class pixelBin {
public:
    std::map<std::pair<int, int>, std::vector<int>> bin;
};

class trainTemplateData {
public:
    std::vector<cv::Point> template_contour;    //contour ảnh template
    std::vector<int> template_img_size;     //kích thước ảnh template

    trainTemplateData(std::vector<cv::Point> template_contour, std::vector<int> template_img_size) {
        this->template_contour = template_contour;
        this->template_img_size = template_img_size;
    }
};

class errorPoint
{
public:
    cv::Point2f point;
    double distance;

    errorPoint(cv::Point2f point, double distance) {
        this->point = point;
        this->distance = distance;
    }
/*
    cv::Point2f getPoint() {
        return point;
    }

    double getDistance() {
        return distance;
    }*/
};

class compareContourResult{
public:
    std::vector<cv::Point> converted_template_contour; //contour ảnh mẫu đã convert toạ độ
    std::vector<cv::Point> target_contour;  // contour ảnh input
    std::vector<errorPoint> errorPoint_in_convertedTempContour;    //Danh sách: toạ độ điểm lỗi + khoảng cách trên đường contour mẫu đã convert
    std::vector<errorPoint> errorPoint_in_targetContour;        //Danh sách: toạ độ điểm lỗi + khoảng cách trên đường contour input

    compareContourResult(std::vector<cv::Point> converted_template_contour, std::vector<cv::Point> target_contour, std::vector<errorPoint> temp_output, std::vector<errorPoint> targ_output) {
        this->converted_template_contour = converted_template_contour;
        this->target_contour = target_contour;
        this->errorPoint_in_convertedTempContour = temp_output;
        this->errorPoint_in_targetContour = targ_output;
    }
};



// Class for preprocessing image
class ImageProcessingOptions {
public:
    virtual ~ImageProcessingOptions() {}
    virtual void apply(const cv::Mat& src, cv::Mat& dst) const = 0;
};

/*class BaseOptions : public ImageProcessingOptions {
public:
    void apply(const cv::Mat& src, cv::Mat& dst) const override {
        dst = src.clone();
    }
};*/

// Subclass Canny
class CannyOptions : public ImageProcessingOptions {
public:
    int lowerValue;
    int higherValue;

    CannyOptions(int lower, int higher) : lowerValue(lower), higherValue(higher) {}

    void apply(const cv::Mat& src, cv::Mat& dst) const override {
        cv::Canny(src, dst, lowerValue, higherValue);
    }
};

// Subclass Threshold
class ThresholdOptions : public ImageProcessingOptions {
public:
    int threshValue;

    ThresholdOptions(int thresh) : threshValue(thresh) {}

    void apply(const cv::Mat& src, cv::Mat& dst) const override {
        cv::threshold(src, dst, threshValue, 255, cv::THRESH_BINARY_INV);
    }
};

/*class OtsuOptions : public ImageProcessingOptions {
public:
    void apply(const cv::Mat& src, cv::Mat& dst) const override {
        cv::Mat otsu_img;
        cv::threshold(src, otsu_img, 0, 255, cv::THRESH_OTSU);
        cv::bitwise_not(otsu_img, dst);
    }
};*/

void processing_image(cv::Mat& image, cv::Mat& processedImage, ImageProcessingOptions& options) {
    options.apply(image, processedImage);
}
/*
class PreprocessImage {
public:
    cv::Mat image;
    int threshValue, lowValue, highValue;

    PreprocessImage(int threshValue, int lowValue, int highValue) {
        this->threshValue = threshValue;
        this->lowValue = lowValue;
        this->highValue = highValue;
    }

    cv::Mat BaseOption_() {
        cv::Mat processed_img;
        processed_img = image.clone();
        return processed_img;
    }

    cv::Mat ThresholdManualOption_() {
        cv::Mat processed_img;
        cv::threshold(image, processed_img, threshValue, 255, cv::THRESH_BINARY_INV);
        return processed_img;
    }

    cv::Mat ThresholdAutoOption_() {
        cv::Mat processed_img;
        cv::Mat otsu_img;
        cv::threshold(image, otsu_img, 0, 255, cv::THRESH_OTSU);
        cv::bitwise_not(otsu_img, processed_img);
        return processed_img;
    }

    cv::Mat CannyOption_() {
        cv::Mat processed_img;
        cv::Canny(image, processed_img, lowValue, highValue);
        return processed_img;
    }
};

enum binaryOption { baseOption, thresholdManualOption, thresholdAutoOption, cannyOption };
binaryOption option;

cv::Mat preprocess_image(cv::Mat image, PreprocessImage PI, binaryOption option) {
    PI.image = image;
    cv::Mat processed_img;
    switch (option) {
    case thresholdManualOption:
        processed_img = PI.ThresholdManualOption_();
        break;
    case thresholdAutoOption:
        processed_img = PI.ThresholdAutoOption_();
        break;
    case cannyOption:
        processed_img = PI.CannyOption_();
        break;
    default:
        processed_img = PI.BaseOption_();
        break;
    }
    
    return processed_img;
}
*/

// Define function - extract contour and divide points into bins
std::vector<cv::Point> extract_contour(cv::Mat image, ImageProcessingOptions& option) {
    // Process input_image to binary image with white object and black background
    //cv::Mat processed_img = preprocess_image(image, PI, option);
    cv::Mat processed_img;
    processing_image(image, processed_img, option);

    // Use function findContours in OpenCV to extract contours of bin_img
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(processed_img, contours, cv::RETR_TREE, cv::CHAIN_APPROX_NONE);

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



pixelBin devide_bin(std::vector<cv::Point> contour, int stride) {
    // Divide points in contour into bins
    pixelBin pibin;
    for (int idx = 0; idx < contour.size(); ++idx) {
        auto& point = contour[idx];
        pibin.bin[std::make_pair(point.x / stride, point.y / stride)].push_back(idx);
    }
    return pibin;
}



// Define trainTemplate function:
trainTemplateData trainTemplate(const cv::Mat& temp_img, ImageProcessingOptions& option) {
    std::vector<cv::Point> temp_contour = extract_contour(temp_img, option);
    int height = temp_img.rows;
    int width = temp_img.cols;
    std::vector<int> temp_size{ width, height };
    trainTemplateData trainData(temp_contour, temp_size);

    return trainData;
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
compareContourResult compareContour(const std::vector<cv::Point>& temp_contour, std::vector<int> temp_size, const cv::Mat& targ_img, std::vector<cv::Point2i> rect, int distance_thresh, ImageProcessingOptions& option) {
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
    std::vector<cv::Point> targ_contour = extract_contour(targ_img, option);
    pixelBin temp_bin = devide_bin(converted_temp_contour, patch_size),
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
            if (temp_bin.bin.count(k) > 0) {
                idx_list.insert(idx_list.end(), temp_bin.bin[k].begin(), temp_bin.bin[k].end());
            }
        }

        if (idx_list.size() != 0) {     // Nếu list không rỗng, ta tính toán với các điểm trong list
            double min_distance = std::numeric_limits<double>::max();
            
            for (const auto& idx : idx_list) {
                double distance = std::abs(converted_temp_contour[idx].x - point.x) + std::abs(converted_temp_contour[idx].y - point.y);
                min_distance = std::min(min_distance, distance);
            }
            if (min_distance > distance_thresh) {
                errorPoint ER(cv::Point2f(point), min_distance);
                targ_output.push_back(ER);
            }
        }
        else {        // Nếu list rỗng, ta đưa mức khoảng cách về ngưỡng lỗi
            errorPoint ER(cv::Point2f(point), distance_thresh + 1);
            targ_output.push_back(ER);
        }
    }

    // Calculate distance from coverted template contour to target contour
    for (const cv::Point& point : converted_temp_contour) {
        int x = point.x / patch_size, y = point.y / patch_size;
        int k1 = (point.x % patch_size >= distance_thresh) ? 1 : -1, k2 = (point.y % patch_size >= distance_thresh) ? 1 : -1;

        std::vector<std::pair<int, int>> key_vec{ {x, y}, {x + k1, y}, {x, y + k2}, {x + k1, y + k2} };
        std::vector<int> idx_list;
        for (const auto& k : key_vec) {
            if (targ_bin.bin.count(k) > 0) {
                idx_list.insert(idx_list.end(), targ_bin.bin[k].begin(), targ_bin.bin[k].end());
            }
        }
        
        if (idx_list.size() != 0) {
            double min_distance = std::numeric_limits<double>::max();
            for (const auto& idx : idx_list) {
                double distance = std::abs(targ_contour[idx].x - point.x) + std::abs(targ_contour[idx].y - point.y);
                min_distance = std::min(min_distance, distance);
            }
            if (min_distance > distance_thresh) {
                errorPoint ER(cv::Point2f(point), min_distance);
                targ_output.push_back(ER);
            }
        }
        else {        // Nếu list rỗng, ta đưa mức khoảng cách về ngưỡng lỗi
            errorPoint ER(cv::Point2f(point), distance_thresh + 1);
            targ_output.push_back(ER);
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << elapsed.count() << std::endl;

    compareContourResult result(converted_temp_contour, targ_contour, temp_output, targ_output);

    return result;    
}


// KD-Tree
cv::Mat convertContour(std::vector<cv::Point> contour) {
    std::vector<cv::Point2f> points;
    for (const auto& point : contour) {
        points.push_back(cv::Point2f(point.x, point.y));
    }
    return cv::Mat(points).reshape(1).clone();
}
compareContourResult compareByKDTree(const std::vector<cv::Point>& temp_contour, std::vector<int> temp_size, const cv::Mat& targ_img, std::vector<cv::Point2i> rect, int distance_thresh, ImageProcessingOptions& option) {
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

    std::vector<cv::Point> targ_contour = extract_contour(targ_img, option);
    cv::Mat temp_contour_mat = convertContour(converted_temp_contour), targ_contour_mat = convertContour(targ_contour);

    //Calculate distance and give defect position
    std::vector<errorPoint> targ_output, temp_output;

    // Calculate distance from target contour to template contour
    float distance;

    cv::flann::Index flannIndex_targ(temp_contour_mat, cv::flann::KDTreeIndexParams(2));
    cv::Mat targ_index(targ_contour_mat.rows, 1, CV_32S), targ_dist(targ_contour_mat.rows, 1, CV_32S);
    flannIndex_targ.knnSearch(targ_contour_mat, targ_index, targ_dist, 1, cv::flann::SearchParams(64));
    for (int i = 0; i < targ_index.rows; i++) {
        distance = sqrt(targ_dist.at<float>(i, 0));
        if (distance > distance_thresh) {
            errorPoint ER(targ_contour_mat.at<cv::Point2f>(i, 0),distance);
            targ_output.push_back(ER);
        }
    }

    // Calculate distance from coverted template contour to target contour
    cv::flann::Index flannIndex_temp(targ_contour_mat, cv::flann::KDTreeIndexParams(2));
    cv::Mat temp_index(temp_contour_mat.rows, 1, CV_32S), temp_dist(temp_contour_mat.rows, 1, CV_32S);
    flannIndex_temp.knnSearch(temp_contour_mat, temp_index, temp_dist, 1, cv::flann::SearchParams(64));
    for (int i = 0; i < temp_index.rows; i++) {
        distance = sqrt(temp_dist.at<float>(i, 0));
        if (distance > distance_thresh) {
            errorPoint ER(temp_contour_mat.at<cv::Point2f>(i, 0), distance);
            temp_output.push_back(ER);
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << elapsed.count() << std::endl;

    compareContourResult result(converted_temp_contour, targ_contour, temp_output, targ_output);

    return result;
}