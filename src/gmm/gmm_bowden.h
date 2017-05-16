#ifndef GMM_BOWDEN_H
#define GMM_BOWDEN_H

#include <opencv2/opencv.hpp>
#include <Eigen/Geometry>

using namespace cv;

namespace GMM_BOWDEN
{
static const int defaultNMixtures = 5;
static const int defaultHistory = 200;
static const double defaultBackgroundRatio = 0.7;
static const double defaultVarThreshold = 2.5*2.5;
static const double defaultNoiseSigma = 30*0.5;
static const double defaultInitialWeight = 0.05;

struct MixData
{
    float sortKey;
    float weight;
    Vec3f mean;
    Vec3f var;
};

class BGModelMOG
{

public:
    BGModelMOG();
    void process(Mat &image, Mat &fgmask, Mat &background,float learningRate=-1);
    void process(Mat &image, Mat &depth, Eigen::Affine3d &transform,
                 Mat &fgmask, Mat &background, float learningRate=-1) {}
private:
protected:
    Size frameSize;
    int frameType;
    Mat bgmodel;
    Mat prev_img;
    int nframes;
    int history;
    int nmixtures;
    float varThreshold;
    float backgroundRatio;
    float noiseSigma;
    float initialWeight;

    void initialize(Size _frameSize, int _frameType);
    void updateModel(Mat &image, Mat &fgmask, Mat &bgimg, float learningRate);
};

}

#endif // GMM_BOWDEN_H
