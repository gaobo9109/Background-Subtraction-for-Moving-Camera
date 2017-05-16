#ifndef BGMODELMOGRGB_H
#define BGMODELMOGRGB_H

#include <Eigen/Geometry>
#include <opencv2/opencv.hpp>
#include <bgs_util/image_transformer.h>

namespace BGS_LB
{
static const int defaultNMixtures = 3;
static const float defaultBackgroundRatio = 0.5f;
static const float defaultVarThreshold = 2.5f;
static const float defaultNoiseSigma = 30*0.5;
static const float defaultInitialVar = 50.f;
static const float defaultInitialWeight = 0.05;
static const float defaultLearningRate = 0.001f;

struct GMM
{
    float sortKey;
    float weight;
    cv::Vec3f mean;
    cv::Vec3f var;
};

class BGModelMog
{
public:
    BGModelMog();
    ~BGModelMog(){}

    void initialize(cv::Mat img_in);
    void getBackgroundImage(cv::Mat bgImg) const;
    void process(cv::Mat &img_in, cv::Mat &foreground, cv::Mat &background);
    void process(cv::Mat &img_in, cv::Mat &depth_in, Eigen::Affine3d &transform,Intrinsic &intrinsic, cv::Mat &foreground);


private:
    cv::Size frameSize;
    int frameType;
    int nmixtures;
    float varThreshold;
    float backgroundRatio;
    float initialVar;
    float learningRate;

    std::vector<GMM> bgmodel;
    std::vector<int> bgmodeUsed;

//    GMM* bgmodel;
//    int* bgmodeUsed;

    cv::Mat bgmodelMean;
    cv::Mat prev_img;

    void updateModel(cv::Mat &img_in, cv::Mat &fgmask);
    void updateModel(cv::Mat &img_in, cv::Mat &depth_in, Eigen::Affine3d &transform, Intrinsic &intrinsic, cv::Mat &dst);
    void printModel();
};

} //BGS

#endif
