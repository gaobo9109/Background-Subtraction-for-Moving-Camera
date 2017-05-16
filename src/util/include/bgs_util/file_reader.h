#ifndef FILE_READER_H
#define FILE_READER_H

#include <opencv2/opencv.hpp>
#include <Eigen/Geometry>

class FileReader
{
private:
    std::string data_dir;
    bool img_grey_scale;

public:
    FileReader(std::string dir);
    void readRGB(std::vector<std::string> &rgb_list);
    void readRGBD(std::vector<std::string> &rgb_list, std::vector<std::string> &depth_list);
    void readRGBDWithGroundTruth(std::vector<std::string> &rgb_list, std::vector<std::string> &depth_list,
                                 std::vector<Eigen::Affine3d> &pose_list);
};

#endif // FILE_READER_H
