#include <fstream>
#include <bgs_util/file_reader.h>

FileReader::FileReader(std::string dir)
{
    data_dir = dir;
}

void FileReader::readRGB(std::vector<std::string> &rgb_list)
{
    std::vector<cv::String> filenames;
    cv::String img_folder(data_dir + "rgb");
    cv::glob(img_folder,filenames,false);

    for(auto &str : filenames)
    {
        rgb_list.push_back(str);
    }
}

void FileReader::readRGBD(std::vector<std::string> &rgb_list, std::vector<std::string> &depth_list)
{
    std::ifstream file(data_dir + "rgb_depth.txt");
    std::string line;

    while(std::getline(file,line))
    {
        std::string rgb_str,depth_str;
        std::istringstream img_pair(line);
        img_pair >> rgb_str >> depth_str;
        rgb_list.push_back(data_dir + rgb_str);
        depth_list.push_back(data_dir + depth_str);
    }
}

void FileReader::readRGBDWithGroundTruth(std::vector<std::string> &rgb_list, std::vector<std::string> &depth_list,
                                    std::vector<Eigen::Affine3d> &pose_list)
{
    std::ifstream img_file(data_dir + "rgb_depth.txt");
    std::ifstream pose_file(data_dir + "rgb_pose.txt");

    std::string line,line1;

    while(std::getline(img_file,line) && std::getline(pose_file,line1))
    {
        std::istringstream img_pair(line);
        std::istringstream quaternion(line1);

        std::string rgb_str,depth_str,rgb_str1;
        float tx,ty,tz,qx,qy,qz,qw;

        img_pair >> rgb_str >> depth_str;
        quaternion >> rgb_str1 >> tx >> ty >> qx >> qy >> qz >> qw;

        if(rgb_str.compare(rgb_str1) != 0) continue;

        rgb_list.push_back(data_dir + rgb_str);
        depth_list.push_back(data_dir + depth_str);

        Eigen::Matrix4d pose_matrix;
        pose_matrix.setZero();
        Eigen::Matrix3d r(Eigen::Quaterniond(qw,qx,qy,qz));
        pose_matrix.block<3,3>(0,0) = r;
        pose_matrix.block<4,1>(0,3) << tx,ty,tz,1;
        Eigen::Affine3d pose;
        pose.matrix() = pose_matrix;
        pose_list.push_back(pose);

    }
}
