#include "pch.h"
#include <iostream>
#include "InspectContour.h"
#include "DistanceSimd.h"
#include "EdgeSubPix.h"


namespace sv {
    /// <summary>
/// Hàm tìm danh sách các tọa độ các điểm contour so với hệ tọa độ ảnh truyền vào
/// </summary>
/// <param name="image"></param>
/// <param name="min_length_contour"></param>
/// <param name=""></param>
/// <returns></returns>
    std::vector<cv::Point2f> extractContour(cv::Mat image, InspectContourOptions options) {
        // Lọc canny 
        cv::Mat cannyDetectionImg;
        cv::Canny(image, cannyDetectionImg, options.LowerThresh, options.UpperThresh);

        // Tìm contour
        vector<vector<cv::Point>> contours;
        cv::findContours(cannyDetectionImg, contours, cv::RETR_TREE, cv::CHAIN_APPROX_NONE);

        // Lọc ra các contour thỏa mãn điều kiện độ dài
        std::vector<cv::Point> contour;
        for (const std::vector<cv::Point>& cnt : contours) {
            if (cv::arcLength(cnt, true) >= options.MinLengthContour) {
                contour.insert(contour.end(), cnt.begin(), cnt.end());
            }
        }
        // Nếu sử dụng chế độ subpixel thì nội suy tiếp trả về các tọa độ
        double alpha = 1.0; // ??
        std::vector<cv::Point2f> templateContour;
        if (options.SubPixelMode == true) {
            vector<vector<cv::Point>> vectorContour{ contour };
            templateContour = InPix2SubPix(image, vectorContour, alpha);
        }
        else {
            for (cv::Point point : contour) {
                templateContour.push_back(cv::Point2f(point));
            }
        }

        // Test in template contour
        cv::Mat img_contours;
        cv::cvtColor(image, img_contours, cv::COLOR_GRAY2BGR);
        for (const auto point : templateContour) {
            cv::circle(img_contours, (Point)point, 2, cv::Scalar(0, 255, 0), -1);
        }
        cv::namedWindow("Template", cv::WINDOW_NORMAL);
        cv::imshow("Template", img_contours);
        return templateContour;
    }

    /// <summary>
    /// Hàm tìm danh sách các tọa độ các điểm contour từ vùng ROI so với hệ tọa độ ảnh truyền vào
    /// </summary>
    /// <param name="image"></param>
    /// <param name="roi"></param>
    /// <param name="options"></param>
    /// <returns></returns>
    std::vector<cv::Point2f> extractContourFromRoi(cv::Mat inputImage, cv::RotatedRect roi, InspectContourOptions options) {
        cv::Rect boundingRectROI = roi.boundingRect();
        // Nếu vùng này vượt quá ảnh, thu gọn hình
        if (boundingRectROI.x < 0) { boundingRectROI.x = 0; }
        if (boundingRectROI.y < 0) { boundingRectROI.y = 0; }
        if (boundingRectROI.x + boundingRectROI.width > inputImage.cols) { boundingRectROI.width = inputImage.cols - boundingRectROI.x; }
        if (boundingRectROI.y + boundingRectROI.height > inputImage.rows) { boundingRectROI.height = inputImage.rows - boundingRectROI.y; }
        if (roi.angle == 0) {
            if (boundingRectROI.width != 0 && boundingRectROI.height != 0) {
                // crop lấy ảnh vùng xử lý và clone dữ liệu mới ????
                cv::Mat imgCrop = inputImage(boundingRectROI).clone();
                if (imgCrop.channels() > 1) {
                    cvtColor(imgCrop, imgCrop, cv::COLOR_BGR2GRAY);
                }
                vector<cv::Point2f> contour = extractContour(imgCrop, options);

                // Có tọa độ contour trong hệ tọa độ ảnh crop -> chuyển sang hệ tọa độ ảnh gốc
                for (int i = 0; i < contour.size(); i++) {
                    contour[i].x += boundingRectROI.tl().x;
                    contour[i].y += boundingRectROI.tl().y;
                }
                return contour;
            }
            else {
                return vector<cv::Point2f>();
            }
        }
        else {
            // Offset lại tọa độ của 4 đỉnh RotatedBoundingbox so với hệ tọa độ Template
            Point2f srcPoints[4];
            roi.points(srcPoints);            
            // Tọa độ 4 đỉnh của Boundingbox đứng cần xoay về thẳng đứng (4 góc của ảnh),  (0,0) tương ứng vị trí đỉnh nào của rotated rect thì đỉnh đó là gốc ảnh
            Point2f dstPoints[4] = { {0, roi.size.height}, {0, 0}, {roi.size.width, 0}, {roi.size.width, roi.size.height} };
            // Xoay và crop ảnh
            Mat matrix = getPerspectiveTransform(srcPoints, dstPoints);
            Mat imgCrop; // Ảnh này tham chiếu đến cùng ảnh gốc hay copy data mới sau khi dùng warpPerspective ???
            warpPerspective(inputImage, imgCrop, matrix, Point(roi.size.width, roi.size.height));
            if (imgCrop.channels() > 1) {
                cvtColor(imgCrop, imgCrop, cv::COLOR_BGR2GRAY);
            }
            vector<cv::Point2f> cnt = extractContour(imgCrop, options);
            return cnt;           
        }
    }
    /// <summary>
    /// 
    /// </summary>
    /// <param name="srcPoints"></param>
    /// <param name="center"></param>
    /// <param name="angle"></param>
    /// <param name="offsetPoints"></param>
    /// <returns></returns>
    vector<Point2f> convertCoordinate(vector<Point2f> srcPoints, Point2f center, float angle, Point2f offsetPoints) {
        vector<cv::Point2f> dstPoints;
        for (int i = 0; i < srcPoints.size(); i++) {
            cv::Point2f pointInImgCoord = sv::rotatePoint(srcPoints[i], center, angle, offsetPoints);
            dstPoints.push_back(pointInImgCoord);
        }
        return dstPoints;
    }

    /// <summary>
    /// Duyệt từng tọa độ contour và chia mỗi tọa độ vào các bin, các điểm tring cùng lân cận sẽ được nhóm vào một bin
    /// </summary>
    /// <param name="contour"></param>
    /// <param name="stride">: ???</param>
    /// <returns></returns>
    PixelBin devideBin(vector<cv::Point2f> contour, double stride) {
        // Divide points in contour into bins
        PixelBin pibin;
        for (int idx = 0; idx < contour.size(); ++idx) {
            auto& point = contour[idx];
            // Nếu cùng lân cận <stride> thì các điểm sẽ có cùng giá trị cặp pair<int,int> 
            (pibin.Bin[make_pair((int)(point.x / stride), (int)(point.y / stride))]).push_back(idx);
        }
        return pibin;
    }

    /// <summary>
    /// Hàm train ảnh template từ ảnh lớn đầu vào và vùng ROI trên giao diện gửi xuống
    /// </summary>
    /// <param name="templateImage"></param>
    /// <param name="options"></param>
    /// <returns></returns>
    TrainTemplateData trainTemplate(cv::Mat inputImage, cv::RotatedRect roi, InspectContourOptions options) {
        return TrainTemplateData(extractContourFromRoi(inputImage, roi, options));
    }

    /// <summary>
    /// Calculate distance of 2 contour
    /// </summary>
    /// <param name="cnt_src"></param>
    /// <param name="cnt_dst"></param>
    /// <param name="bin_dst"></param>
    /// <param name="distance_thresh"></param>
    /// <returns></returns>
    vector<ErrorPoint> calDisOfContour(std::vector<cv::Point2f> cnt_src, std::vector<cv::Point2f> cnt_dst, PixelBin bin_dst, double distance_thresh) {

        vector<ErrorPoint> output;
        double patch_size = 2 * distance_thresh;

        for (const cv::Point2f& point : cnt_src) {
            int x = (int)(point.x / patch_size), y = (int)(point.y / patch_size);   // Lẩy chỉ số của bin chứa điểm đang xét
            int k1 = ((point.x / patch_size - (int)(point.x / patch_size)) >= 0.5) ? 1 : -1, k2 = ((point.y / patch_size - (int)(point.y / patch_size)) >= 0.5) ? 1 : -1; // Lấy chỉ số của 3 bin lân cận cần xét

            std::vector<std::pair<int, int>> key_vec{ {x, y}, {x + k1, y}, {x, y + k2}, {x + k1, y + k2} };
            // Tạo list gồm các index trong các bin
            vector<int> idx_list;
            for (const auto& k : key_vec) {
                if (bin_dst.Bin.count(k) > 0) {
                    idx_list.insert(idx_list.end(), bin_dst.Bin[k].begin(), bin_dst.Bin[k].end());
                }
            }
            if (idx_list.size() != 0) {     // Nếu list không rỗng, ta tính toán với các điểm trong list
                double min_distance = 0;

                for (const auto& idx : idx_list) {
                    //double distance = sqrt( pow((cnt_dst[idx].x - point.x),2) + pow((cnt_dst[idx].y - point.y),2));
                    double distance = std::abs(cnt_dst[idx].x - point.x) + std::abs(cnt_dst[idx].y - point.y);
                    min_distance = min(min_distance, distance);
                }
                if (min_distance > distance_thresh) {
                    ErrorPoint ER(point, min_distance);
                    output.push_back(ER);
                }
            }
            else { // Nếu list rỗng, ta đưa mức khoảng cách về ngưỡng lỗi
                ErrorPoint ER(point, distance_thresh + 1);
                output.push_back(ER);
            }
        }
        return output;
    }

    /// <summary>
    /// 
    /// </summary>
    /// <param name="cnt_src"></param>     
    /// <param name="cnt_dst"></param>
    /// <param name="bin_dst"></param>
    /// <param name="distance_thresh"></param>
    /// <returns></returns>
    vector<ErrorPoint> calDisOfContourSse2(vector<cv::Point2f> srcContour, vector<cv::Point2f> dstContour, PixelBin bin_dst, double distanceThresh) {

        vector<ErrorPoint> output;
        double patchSize = 2 * distanceThresh;

        for (const cv::Point2f& point : srcContour) {
            int xBin = (int)(point.x / patchSize), yBin = (int)(point.y / patchSize);              // Tại điểm đang xét xem nó ở bin nào ?
            int vicinityX = ((point.x / patchSize - (int)(point.x / patchSize)) >= 0.5) ? 1 : -1;  // Lấy chỉ số của 3 bin lân cận cần xét
            int vicinityY = ((point.y / patchSize - (int)(point.y / patchSize)) >= 0.5) ? 1 : -1;

            vector<pair<int, int>> key_vec{ {xBin, yBin}, {xBin + vicinityX, yBin}, {xBin, yBin + vicinityY}, {xBin + vicinityX, yBin + vicinityY} };
            // Tạo list gồm index của các điểm trên vector contour cần kiểm tra với điểm đang xét
            vector<int> idx_list;
            for (const auto& k : key_vec) {
                if (bin_dst.Bin.count(k) > 0) {
                    idx_list.insert(idx_list.end(), bin_dst.Bin[k].begin(), bin_dst.Bin[k].end());
                }
            }
            int length = idx_list.size();
            if (length != 0) {     // Nếu list không rỗng, ta tính toán với các điểm trong list
                float minDistance = 0;

                float* x_i = new float[length];
                float* y_i = new float[length];
                for (size_t i = 0; i < length; ++i) {
                    x_i[i] = dstContour[idx_list[i]].x;
                    y_i[i] = dstContour[idx_list[i]].y;
                }
                minDistance = L2DistSSE2(point.x, point.y, x_i, y_i, length, minDistance);
                minDistance = sqrt(minDistance);
                delete[] x_i, y_i;
                // Nếu dùng GPU cho 1 thread của work-group tổng hợp kết quả minDistance
                if (minDistance > distanceThresh) {
                    ErrorPoint ER(point, minDistance);
                    output.push_back(ER);
                }
            }
            else { // Nếu list rỗng, ta đưa mức khoảng cách về ngưỡng lỗi
                ErrorPoint ER(point, distanceThresh + 1);
                output.push_back(ER);
            }
        }
        return output;
    }
    /// <summary>
    /// 
    /// </summary>
    /// <param name="temp_contour"></param>
    /// <param name="temp_size"></param>
    /// <param name="targ_img"></param>
    /// <param name="rect"></param>
    /// <param name="distance_thresh"></param>
    /// <param name="min_length_contour"></param>
    /// <param name="option"></param>
    /// <param name="subPixelMode"></param>
    /// <returns></returns>
    InspectContourResult inspectContour(cv::Mat inputImage, cv::RotatedRect roi, TrainTemplateData trainedData, InspectContourOptions options) {
        // Crop ảnh từ vùng ROi và tính tọa độ target contour so với hệ ảnh crop
        vector<cv::Point2f> targetContourInRoi = extractContourFromRoi(inputImage, roi, options);
        // Convert sang hệ tọa độ ảnh gốc tìm kiếm
        cv::Point2f centerOfRotation((float)roi.size.width / 2, (float)roi.size.height / 2);
        vector<cv::Point2f> targetContourInImg = convertCoordinate(targetContourInRoi, centerOfRotation, roi.angle, roi.center);
        // Convert tọa độ contour template sang hệ tọa độ ảnh gốc tìm kiếm dựa vào (center-w/h-angle của Roi)
        vector<cv::Point2f> templateContourInImg = convertCoordinate(trainedData.TemplateContour, centerOfRotation, roi.angle, roi.center);

        // Divide bin
        double patch_size = 2 * options.DistanceThresh;
        PixelBin temp_bin = devideBin(templateContourInImg, patch_size);
        PixelBin targ_bin = devideBin(targetContourInImg, patch_size);

        // Calculate distance from target contour to template contour
        std::vector<ErrorPoint> targ_output = calDisOfContourSse2(targetContourInImg, templateContourInImg, temp_bin, options.DistanceThresh);
        //std::vector<errorPoint> targ_output = calDisOfContour(targ_contour, converted_temp_contour, temp_bin, distance_thresh);

        // Calculate distance from coverted template contour to target contour
        std::vector<ErrorPoint> temp_output = calDisOfContourSse2(templateContourInImg, targetContourInImg, targ_bin, options.DistanceThresh);
        //std::vector<errorPoint> temp_output = calDisOfContour(converted_temp_contour, targ_contour, targ_bin, distance_thresh);

        InspectContourResult result(targetContourInImg, targ_output, temp_output);

        return result;
    }
}

