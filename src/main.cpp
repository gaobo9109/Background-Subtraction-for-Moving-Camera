#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <sstream>
#include <Eigen/Geometry>
//#include "util/warp_image.h"
#include "gmm/BGModelMog.h"
#include "gmm/bgfg_gaussmix2.h"
#include "util/read_lines.h"
#include "rgbd_motion_removal/motion_removal.h"
#include "util/surface_pyramid.h"
#include "util/image_transformer.h"

void read_image_files(cv::String image_folder)
{
    std::vector<cv::String> filenames;
    cv::glob(image_folder,filenames);

    cv::Mat img,foreground,frame;
    int key;

    int count = 0;

    for(int i=0; i<filenames.size(); i++)
    {
        img = cv::imread(filenames[i]);
        cv::cvtColor(img,frame,CV_BGR2GRAY);


//        while(cv::waitKey(0) != 'c');
        cv::waitKey(30);

    }
}

void read_rgbd_with_ground_truth(std::string dir)
{
    std::string output = dir + "output.txt";
    std::string output1 = dir + "output2.txt";
    std::ifstream file(output);
    std::ifstream file1(output1);
    std::string image,trajectory;
    Eigen::Affine3d curr_pose, prev_pose;
    cv::Mat prev_img,rgb,depth,foreground,background,model;
//    Intrinsic intrinsic = {525.0, 525.0, 319.5, 239.5};
    Intrinsic intrinsic = {520.9, 521.0, 325.1, 249.7};
    ImageTransformer transformer(intrinsic);
    BGS_ZIKOVIC::BGModelMog gmm(intrinsic);


    while(std::getline(file,image) && std::getline(file1,trajectory))
    {
        bool success = readLines(dir, image,trajectory,curr_pose,rgb,depth);
        if(!success) continue;

        if(prev_img.empty())
        {
//            transformer.setImageDimension(rgb.cols,rgb.rows);
            prev_pose = curr_pose;
//            model = rgb.clone();
//            prev_img = rgb.clone();
//            continue;
        }

        Eigen::Affine3d forward_transform = curr_pose * prev_pose.inverse();
        Eigen::Affine3d reverse_transform = prev_pose * curr_pose.inverse();

        gmm.process(rgb,depth,reverse_transform,foreground,background,0.25);

//        transformer.warp3D(model,rgb,depth,reverse_transform);
//        cv::Mat residue,residue1;
//        cv::absdiff(model,rgb,residue);
//        cv::absdiff(prev_img,rgb,residue1);
//        cv::imshow("residue",residue);
//        cv::imshow("residue1",residue1);
//        cv::imshow("previous",prev_img);
//        cv::imshow("current",rgb);
//        cv::imshow("model",model);


        prev_pose = curr_pose;
        prev_img = rgb;

        cv::imshow("foreground",foreground);
        cv::imshow("background",background);
        cv::imshow("original",rgb);

//        cv::waitKey(30);
        while(cv::waitKey(0) != 'c');
    }

}

void read_video_files(std::string video_file)
{
    cv::VideoCapture capture(video_file);
    BGS_ZIKOVIC::BGModelMog gmm;
    cv::Mat img,foreground,background,frame;

    while(capture.isOpened())
    {
        capture >> img;
        if(img.empty())break;

//        cv::cvtColor(img,frame,CV_BGR2GRAY);
        gmm.process(img,foreground,background);
//        gmm.process(img,foreground,background);

        cv::imshow("foreground",foreground);
        cv::imshow("background",background);
        cv::imshow("original",img);
        while(cv::waitKey(0) != 'c');
    }

}

void read_rgbd(std::string dir)
{
    std::ifstream file(dir+"output.txt");
    std::string img_pair;
    MotionRemoval remover;

    while(std::getline(file,img_pair))
    {
        std::string rgb_img,depth_img;
        std::istringstream iss(img_pair);
        iss >> rgb_img >> depth_img;
        cv::Mat rgb = cv::imread(dir+rgb_img,CV_LOAD_IMAGE_GRAYSCALE);
        cv::Mat depth_temp = cv::imread(dir+depth_img,CV_LOAD_IMAGE_ANYDEPTH);
        cv::Mat depth;
        SurfacePyramid::convertRawDepthImageSse(depth_temp, depth, 0.0002);

        remover.process(rgb,depth);
//        cv::imshow("rgb",rgb);
//        cv::waitKey(30);
        while(cv::waitKey(0) != 'c');
    }
}

int main()
{

//    read_image_files("/home/gaobo/dataset/fr2_desk/rgb/");
//    read_video_files("/home/gaobo/Video_003/walking.avi");
    read_rgbd_with_ground_truth("/home/gaobo/dataset/fr2_desk/");
//    read_rgbd("/home/gaobo/dataset/fr2_desk/");

    return 0;
}
