#ifndef BGFG_GAUSSMIX2_H
#define BGFG_GAUSSMIX2_H

#include <opencv2/opencv.hpp>
#include <Eigen/Geometry>
#include <bgs_util/image_transformer.h>
#include <memory>

using namespace cv;

namespace GMM_ZIKOVIC
{
static const int defaultHistory2 = 500; // Learning rate; alpha = 1/defaultHistory2
static const float defaultVarThreshold2 = 4.0f*4.0f;
static const int defaultNMixtures2 = 5; // maximal number of Gaussians in mixture
static const float defaultBackgroundRatio2 = 0.9f; // threshold sum of weights for background test
static const float defaultVarThresholdGen2 = 3.0f*3.0f;
static const float defaultVarInit2 = 15.0f; // initial variance for new components
static const float defaultVarMax2 = 5*defaultVarInit2;
static const float defaultVarMin2 = 4.0f;

static const float defaultfCT2 = 0.05f; // complexity reduction prior constant 0 - no reduction of number of components

struct GMM
{
    float weight;
    float var;
};

class BGModelMOG
{
public:
    BGModelMOG();
    void process(Mat &image, Mat &fgmask, Mat &background,float learningRate=-1);
    void process(Mat &image, Mat &depth, Eigen::Affine3d &transform,
                 Mat &fgmask, Mat &background, float learningRate=-1);


private:
    void initialize(Size _frameSize, int _frameType);
    void initialize(Size _frameSize, int _frameType, Intrinsic &intrinsic);
    void updateModel(Mat &image, Mat &fgmask, Mat &bgimg, float learningRate);
    void warpModel(Mat &image, Mat &depth, Mat &fgmask, Eigen::Affine3d &transform, float learningRate);
    void updateWarpedModel(Mat &image, Mat &fgmask, Mat &bgimg, float learningRate);
    void showBackgroundImage();
    void showDifferenceImage(Mat &img1, Mat &img2);

    Mat bgmodel;
    Mat prev_img;
    Mat bgmodelUsedModes;//keep track of number of modes per pixel
    Mat update_mask;

    GMM* gmm0;
    float* mean0;
    uchar* modesUsed0;

    std::shared_ptr<ImageTransformer> transformer;

    Size frameSize;
    int frameType;

    int nframes;
    int history;
    int nmixtures;
    float varThreshold;
    float backgroundRatio;

    float varThresholdGen;
    float fVarInit;
    float fVarMin;
    float fVarMax;



};

};

#endif // BGFG_GAUSSMIX2_H
