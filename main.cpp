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

namespace fs = std::filesystem;


int main() {
    double threshold = 215;
    int distance_thresh = 15;

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

    std::vector<cv::Mat> images;
    for (std::string name : list_image) {
        images.push_back(cv::imread(name, 0));
    };

    std::cout << "Enter the index of the template image: ";
    int idxOfTemp;
    std::cin >> idxOfTemp;

    
    std::vector<cv::Point> temp_contour;
    std::map<std::pair<int, int>, std::vector<int>> temp_bin;
    tie(temp_contour, temp_bin) = trainTemplate(images[idxOfTemp], threshold, distance_thresh);

    std::vector<std::vector<std::vector<cv::Point2f>>> pos_list;
    for (size_t i = 0; i < list_image.size(); ++i) {
        if (i != idxOfTemp) {
            std::vector<cv::Point2f> pos_1, pos_2;

            RectangleRoi ROI(200, 200, 500, 600, 20);
            std::vector<cv::Point2d> rec = ROI.getVertices();
            //std::vector<std::pair<cv::Point2f, double>> pos_1, pos_2;
            //std::vector<cv::Point2d> rec = { cv::Point2d(100, 500), cv::Point2d(500, 100), cv::Point2d(900, 500), cv::Point2d(500, 900) };
            tie(pos_1, pos_2) = compareContour(temp_contour, temp_bin, images[i], rec, threshold, distance_thresh);
            pos_list.push_back({ pos_1, pos_2 });
        }
    }
    std::cout << "Enter: ";
    int k ;
    std::cin >> k;
    std::cout << pos_list[k][0] << pos_list[k][1] << "\n";

    while (k >= 0) {
        std::cin >> k;
        std::cout << pos_list[k][0] << pos_list[k][1] << "\n";
    }

    return 0;
};
