#ifndef GMM_WITH_DEPTH_H
#define GMM_WITH_DEPTH_H

#include <opencv2/opencv.hpp>
#include <Eigen/Geometry>

using namespace cv;

namespace GMM_WITH_DEPTH
{
static const int defaultNMixtures = 5;
static const int defaultHistory = 200;
static const double defaultBackgroundRatio = 0.7;
static const double defaultVarThreshold = 2.5*2.5;
static const double defaultVarInit = 15.0f;
static const double defaultInitialWeight = 0.05;

struct ColorModel
{
    float sortKey;
    float weight;
    Vec3f mean;
    Vec3f var;
};

struct DepthModel
{
    float sortKey;
    float weight;
    float mean;
    float var;
};

class BGModelMOG
{
public:
    BGModelMOG();
    void process(Mat &image, Mat &fgmask, Mat &background,float learningRate=-1) {}
    void process(Mat &image, Mat &depth, Mat &fgmask, Mat &background, float learningRate=-1);
    void process(Mat &image, Mat &depth, Eigen::Affine3d &transform,
                 Mat &fgmask, Mat &background, float learningRate=-1);

private:
    Size frameSize;
    int frameType;

    int nframes;
    int history;
    int nmixtures;
    float varThreshold;
    float backgroundRatio;
    float initialVar;
    float initialWeight;

    Mat bgModelColor;
    Mat bgModelDepth;
    Mat prev_img;

    void initialize(Size _frameSize, int _frameType);
    void updateModel(Mat &image, Mat &depth, Mat &fgmask, Mat &bgimg, float learningRate);
    bool processDepth(DepthModel* dm, float currVal, float alpha);
    bool processColor(ColorModel* cm, Vec3f currVal, bool fgDepth, float alpha);
};
}

#endif // GMM_WITH_DEPTH_H
