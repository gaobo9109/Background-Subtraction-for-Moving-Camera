#include "image_transformer.h"

ImageTransformer::ImageTransformer(Intrinsic _intrinsic)
{
    intrinsic = _intrinsic;
}

cv::Mat ImageTransformer::warp2D(cv::Mat &model, cv::Mat &prev, cv::Mat &curr)
{
    assert(!prev.empty() && !curr.empty());
    cv::Size frameSize = prev.size();
    std::vector<cv::Point2f>   prevPts;
    std::vector<cv::Point2f>   nextPts;
    std::vector<unsigned char> status;
    std::vector<float>         error;

    int maxCorners = 100;
    double qualityLevel = 0.01;
    double minDistance = 5;
    cv::Mat mask;
    int blockSize = 7;
    bool useHarrisDetector = false;
    double k = 0.04;

    cv::goodFeaturesToTrack(prev, prevPts, maxCorners, qualityLevel, minDistance, mask, blockSize, useHarrisDetector, k);

    if(prevPts.size() >= 1)
    {
        cv::calcOpticalFlowPyrLK(prev, curr, prevPts, nextPts, status, error, cv::Size(20,20), 5);

        cv::Mat mask;
        std::vector <cv::Point2f> prev_corner2, cur_corner2;

        // weed out bad matches
        for(size_t i=0; i < status.size(); i++) {
            if(status[i]) {
                prev_corner2.push_back(prevPts[i]);
                cur_corner2.push_back(nextPts[i]);
            }
        }

        cv::Mat H = cv::findHomography(prev_corner2,cur_corner2, CV_RANSAC);
        cv::warpPerspective(prev, model, H, frameSize, CV_INTER_LINEAR);
        return H;
    }
}

bool ImageTransformer::transformPoint(int x, int y, double z, int &x_out, int &y_out, Eigen::Affine3d &transform)
{
    double pz = z;
    bool canWarp = false;

    if(std::isfinite(pz))
    {
        double px = (x - intrinsic.ox) * pz / intrinsic.fx;
        double py = (y - intrinsic.oy) * pz / intrinsic.fy;
        Eigen::Vector3d p(px,py,pz);
        Eigen::Vector3d p_warped = transform * p;

        int warped_x = (int) (p_warped(0) * intrinsic.fx / p_warped(2) + intrinsic.ox);
        int warped_y = (int) (p_warped(1) * intrinsic.fy / p_warped(2) + intrinsic.oy);

        if(warped_y >=0 && warped_y < image_height && warped_x >= 0 && warped_x < image_width)
        {
            canWarp = true;
            x_out = warped_x;
            y_out = warped_y;
        }
    }

    return canWarp;
}

void ImageTransformer::warp3D(cv::Mat &model, cv::Mat &rgb, cv::Mat &depth, Eigen::Affine3d &transform)
{
    cv::Mat temp = rgb.clone();
    cv::Mat residue;

    for(int y=0; y<rgb.rows; y++)
    {
        for(int x=0; x<rgb.cols; x++)
        {
            double z = (double)depth.at<float>(y,x);
            int warped_x, warped_y;
            bool canWarp = transformPoint(x,y,z,warped_x,warped_y,transform);
            if(canWarp)
                temp.at<uchar>(y,x) = model.at<uchar>(warped_y,warped_x);
        }
    }

    model = temp;
}
