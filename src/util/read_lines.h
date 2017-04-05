#ifndef READ_LINES_H
#define READ_LINES_H

#include <opencv2/opencv.hpp>
#include <Eigen/Geometry>

bool readLines(std::string &dir, std::string &image, std::string &trajectory,
               Eigen::Affine3d &pose, cv::Mat &rgb_in, cv::Mat &depth_in);


#endif // READ_LINES_H


