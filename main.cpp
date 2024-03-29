#include <opencv2/opencv.hpp>
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/core/core.hpp"
#include <opencv2/flann.hpp>
#include <filesystem>
#include <iostream>
#include <vector>
#include <map>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <sys/stat.h>

#include "function.cpp"
#include "ROI_Cpp/roi.cpp"
#include "EdgesSubPix/EdgesSubPix.cpp"

namespace fs = std::filesystem;

std::pair < std::vector < int >, double >
makeROI(const cv::Mat& temp_img, const cv::Mat& targ_img) {
    cv::Ptr<cv::SiftFeatureDetector> detector;
    detector = cv::SiftFeatureDetector::create();
    std::vector<cv::KeyPoint> keypoints_temp, keypoints_targ;
    cv::Mat descriptors_temp, descriptors_targ;
    cv::BFMatcher bf;

    detector->detectAndCompute(temp_img, cv::noArray(), keypoints_temp, descriptors_temp);
    detector->detectAndCompute(targ_img, cv::noArray(), keypoints_targ, descriptors_targ);

    std::vector<std::vector<cv::DMatch>> knn_matches;
    bf.knnMatch(descriptors_temp, descriptors_targ, knn_matches, 2);
    std::vector<cv::DMatch> good_matches;
    for (const auto& match_pair : knn_matches) {
        if (match_pair[0].distance < 0.75 * match_pair[1].distance) {
            good_matches.push_back(match_pair[0]);
        }
    }
    std::vector<cv::Point2f> points_temp, points_targ;
    for (const auto& m : good_matches) {
        points_temp.push_back(keypoints_temp[m.queryIdx].pt);
        points_targ.push_back(keypoints_targ[m.trainIdx].pt);
    }

    // Ước lượng phép biến đổi Affine
    cv::Mat M = cv::estimateAffinePartial2D(points_temp, points_targ);
    // Trích xuất các giá trị dịch chuyển và góc xoay
    int Dx = round( M.at<double>(0, 2) ); //truc 0x
    int Dy = round( M.at<double>(1, 2) ); //truc Oy
    double Dr = std::atan(M.at<double>(1, 0) / M.at<double>(0, 0));
    if (abs( cos(Dr) - M.at<double>(0, 0) ) < 0.01) { Dr = Dr * 180 / CV_PI; }
    else {Dr = 180 + Dr * 180 / CV_PI;}

    std::vector<int> rect{ Dx, Dy };
    return { rect, Dr };
}

std::vector<cv::Point2i> 
get4POint(int x, int y, int w, int h, double angle) {
    std::vector<cv::Point2i> vts{
            Point2i(x, y),
            Point2i(x + w, y),
            Point2i(x + w, y + h),
            Point2i(x, y + h)
    };
    if (angle == 0) { return vts; };
    double angle_rad = angle * CV_PI / 180;
    double alpha_w = std::atan(w*1.0 / h);
    double alpha_h = std::atan(h*1.0 / w);
    double half_cross = sqrt(w * w + h * h) / 2;
    cv::Point2f center = cv::Point2f( x + half_cross * sin(alpha_w - angle_rad), y + half_cross * cos(alpha_w - angle_rad) );
    vts[1] = cv::Point2i(round(center.x + half_cross * cos(alpha_h - angle_rad)), round(center.y - half_cross * sin(alpha_h - angle_rad)) );
    vts[2] = cv::Point2i(round(2 * center.x - vts[0].x) , round(2 * center.y - vts[0].y) );
    vts[3] = cv::Point2i(round(2 * center.x - vts[1].x) , round(2 * center.y - vts[1].y) );
    return vts;
}


int main() {    
    int distance_thresh = 15;
    //double min_length_contour = 100;
    int threshValue = 215, lowValue = 125, highValue = 215;

    enum binaryOption { thresholdOptions_, cannyOptions_, otsuOptions_ };
    binaryOption b_option = binaryOption(0);
    //PreprocessImage PI(threshValue, lowValue, highValue);
    
    CannyOptions            cannyOption(lowValue, highValue);       // Sử dụng CannyOptions  
    ThresholdOptions        thresholdOption(threshValue);           // Sử dụng ThresholdOptions
    //OtsuOptions             otsuOption();                           // Sử dụng threshOtsu
    //BaseOptions             baseOption();                           // không xử lý gì ảnh
    ImageProcessingOptions& option = cannyOption;

    switch (b_option) {
        case thresholdOptions_:
            ImageProcessingOptions& option = thresholdOption;
            std::cout << "0";
            break;
        case cannyOptions_:
            ImageProcessingOptions& option = cannyOption;
            std::cout << "1";
            break;
        /*case otsuOptions_:
            ImageProcessingOptions& option = otsuOption;
            break;
        default:
            ImageProcessingOptions& option = baseOption;
            break;*/
    }

    std::string path = "C:\\Users\\Admin\\source\\repos\\InspectContour\\NG";
    //std::cout<< "Enter the path of imageset: ";
    //std::cin >> path;
    
    // This structure would distinguish a file from a directory
    struct stat sb;
    std::vector<std::string> list_image;
    for (const auto& entry : fs::directory_iterator(path)) {
        // Converting the path to const char * in the subsequent lines
        std::filesystem::path outfilename = entry.path();
        std::string outfilename_str = outfilename.string();
        const char* img_path = outfilename_str.c_str();

        // Testing whether the path points to a non-directory or not
        if (stat(img_path, &sb) == 0 && !(sb.st_mode & S_IFDIR)) {
            list_image.push_back(img_path);
        }
    }

    // Read image
    std::vector<cv::Mat> images;
    for (std::string name : list_image) {
        images.push_back(cv::imread(name, 0));
    };

    // Subpixel
    /*double alpha = 0;
    int low = 125, high = 215;
    std::vector<Contour> contours;
    std::cout << "Subpixel contour: \n";
    EdgesSubPix(images[0], alpha, low, high, contours);
    std::cout << contours.size();
    for (const Contour& contour : contours) {
        std::cout << contour.points  << "\n";
    }*/

    //ImageProcessingOptions& option = thresholdOption;
    // Train template
    trainTemplateData trainData = trainTemplate(images[15], option);
    std::vector<cv::Point> temp_contour = trainData.template_contour;
    std::vector<int> temp_size = trainData.template_img_size;
    
    //Compare contour
    std::vector<compareContourResult> results;    
    for (size_t i = 0; i < list_image.size(); ++i) {
        //RectangleRoi ROI(r[0], r[1], temp_size[0], temp_size[1], angle);
        //std::vector<cv::Point2d> rec = ROI.getVertices();
        std::vector<int> r;
        double angle; 
        tie(r, angle) = makeROI(images[15], images[i]);
        std::vector<cv::Point2i> vts = get4POint(r[0], r[1], temp_size[0], temp_size[1], angle);

        compareContourResult result = compareContour(temp_contour, temp_size, images[i], vts, distance_thresh, option);
        results.push_back(result);
    }

    // Print results
    int k = 0;
    while (k >= 0) {
        std::cout << "Enter: ";
        std::cin >> k;
        std::vector<errorPoint> temp_pos = results[k].errorPoint_in_convertedTempContour, targ_pos = results[k].errorPoint_in_targetContour;
        if (temp_pos.empty() && targ_pos.empty()) {
            std::cout << "No error" << "\n";
        }
        else {
            for (errorPoint er : temp_pos) {
                std::cout << er.point << " - " << er.distance << "\n";
            }
            for (errorPoint er : targ_pos) {
                std::cout << er.point << " - " << er.distance << "\n";
            }
        }
    }
    
    return 0;
};