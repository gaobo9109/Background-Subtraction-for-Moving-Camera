#include "read_lines.h"
#include "surface_pyramid.h"

bool readLines(std::string &dir, std::string &image, std::string &trajectory,
               Eigen::Affine3d &pose, cv::Mat &rgb_in, cv::Mat &depth_in)
{
    std::istringstream iss(image);
    std::istringstream iss1(trajectory);
    std::string rgb,rgb1,depth;
    float tx,ty,tz,qx,qy,qz,qw;

    iss >> rgb >> depth;
    iss1 >> rgb1 >> tx >> ty >> tz >> qx >> qy >> qz >> qw;

    if(rgb.compare(rgb1) != 0) return false;

//    rgb_in = cv::imread(dir+rgb,CV_LOAD_IMAGE_GRAYSCALE);
    rgb_in = cv::imread(dir+rgb);
    cv::Mat depth_temp = cv::imread(dir+depth,CV_LOAD_IMAGE_ANYDEPTH);
    SurfacePyramid::convertRawDepthImageSse(depth_temp,depth_in,0.0002);

    Eigen::Matrix4d temp;
    temp.setZero();
    Eigen::Matrix3d r(Eigen::Quaterniond(qw,qx,qy,qz));

    temp.block<3,3>(0,0) = r;
    temp.block<4,1>(0,3) << tx,ty,tz,1;

    pose.matrix() = temp;
    return true;
}


