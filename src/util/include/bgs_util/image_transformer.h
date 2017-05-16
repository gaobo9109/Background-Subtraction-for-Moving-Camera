#ifndef IMAGE_TRANSFORMER_H
#define IMAGE_TRANSFORMER_H

#include <Eigen/Geometry>
#include <opencv2/opencv.hpp>

struct Intrinsic
{
    double fx;
    double fy;
    double ox;
    double oy;
};


class ImageTransformer
{
public:
    ImageTransformer(Intrinsic _intrinsic);
    bool transformPoint(int x, int y, double z, int &x_out, int &y_out, Eigen::Affine3d &transform);
    cv::Mat warp2D(cv::Mat &model, cv::Mat &prev, cv::Mat &curr);
    void warp3D(cv::Mat &model, cv::Mat &rgb, cv::Mat &depth,Eigen::Affine3d &transform);
    void warp3DRChannelSwap(cv::Mat &model, cv::Mat &rgb, cv::Mat &depth,Eigen::Affine3d &transform);

    //setters and getters
    Intrinsic getIntrinsic() {return intrinsic;}
    void setIntrinsic(Intrinsic _intrinsic) {intrinsic = _intrinsic;}
    void setImageDimension(int _image_width, int _image_height) {image_width = _image_width; image_height = _image_height;}


private:
    Intrinsic intrinsic;
    int image_width;
    int image_height;

};

#endif // IMAGE_TRANSFORMER_H
