#ifndef __DualSGM__
#define __DualSGM__

#include <opencv2/opencv.hpp>
#include <Eigen/Geometry>
#include "../util/warp_image.h"


class DualSGM
{
public:
    ~DualSGM();
    DualSGM();

    static int num_rows;
    static int num_cols;

    void init(const cv::Mat &first_image);
    void process(cv::Mat &next_frame, cv::Mat &depth, Eigen::Affine3d &transform, Intrinsic intrinsic, cv::Mat &foreground);
    void process(const cv::Mat &next_frame, cv::Mat &foreground);

private:
    void warpBack(cv::Mat &next_frame, cv::Mat &depth,
                  Eigen::Affine3d &transform, Intrinsic &intrinsic);

    void warpForward(Eigen::Affine3d &transform, Intrinsic &intrinsic);
    void warpForward(const cv::Mat &next_frame, const cv::Mat &depth, Eigen::Affine3d &transform, Intrinsic intrinsic);

    cv::Mat prev_frame;
    cv::Mat prev_depth;
    cv::Mat bin_mat;
    cv::Mat app_u_mat;
    cv::Mat app_var_mat;
    cv::Mat can_u_mat;
    cv::Mat can_var_mat;
    cv::Mat app_age_mat;
    cv::Mat can_age_mat;

};

inline void core_dsgm_update(const cv::Mat &next_frame, cv::Mat &bin_mat,
                             cv::Mat &app_u_mat, cv::Mat &app_var_mat,
                             cv::Mat &can_u_mat, cv::Mat &can_var_mat,
                             cv::Mat &app_age_mat, cv::Mat &can_age_mat,
                             int offset, int row_lim);

#endif /* defined(__DualSGM__) */
