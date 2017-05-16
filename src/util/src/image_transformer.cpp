#include <bgs_util/image_transformer.h>

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
    Eigen::Matrix3d K;
    K << intrinsic.fx, 0, intrinsic.ox,
         0, intrinsic.fy, intrinsic.oy,
         0, 0, 1;

    Eigen::Matrix<double, 3, 4> KT = K * transform.matrix().block<3,4>(0,0);

    if(std::isfinite(pz))
    {
        double px = (x - intrinsic.ox) * pz / intrinsic.fx;
        double py = (y - intrinsic.oy) * pz / intrinsic.fy;
        Eigen::Vector4d p(px,py,pz,1);
        Eigen::Vector4d p_warped;
        p_warped.setConstant(1);
        p_warped.head<3>() = KT * p;

        int warped_x = (int)std::round(p_warped(0) / p_warped(2));
        int warped_y = (int)std::round(p_warped(1) / p_warped(2));

//        int warped_x = (int)std::round(p_warped(0) * intrinsic.fx / p_warped(2) + intrinsic.ox);
//        int warped_y = (int)std::round(p_warped(1) * intrinsic.fy / p_warped(2) + intrinsic.oy);

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
    int nchannels = rgb.channels();
    cv::Mat temp = rgb.clone();
    for(int y=0; y<rgb.rows; y++)
    {
        for(int x=0; x<rgb.cols; x++)
        {
            double z = (double)depth.at<float>(y,x);
            int warped_x, warped_y;
            bool canWarp = transformPoint(x,y,z,warped_x,warped_y,transform);
            if(canWarp)
            {
                temp.at<cv::Vec3b>(y,x) = model.at<cv::Vec3b>(y,x);
            }

        }
    }

    model = temp;
}

void ImageTransformer::warp3DRChannelSwap(cv::Mat &model, cv::Mat &rgb, cv::Mat &depth, Eigen::Affine3d &transform)
{
    cv::Mat temp = rgb.clone();
    for(int y=0; y<rgb.rows; y++)
    {
        for(int x=0; x<rgb.cols; x++)
        {
            double z = (double)depth.at<float>(y,x);
            int warped_x, warped_y;
            bool canWarp = transformPoint(x,y,z,warped_x,warped_y,transform);
            if(canWarp)
            {
                cv::Vec3b src = model.at<cv::Vec3b>(y,x);
                cv::Vec3b dst = rgb.at<cv::Vec3b>(y,x);
                dst(2) = src(2);
                temp.at<cv::Vec3b>(y,x) = dst;
            }
        }
    }
    model = temp;
}
