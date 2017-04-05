#include "dual_sgm.h"

/* Pre processing */
#define GAUSSIAN_SIZE 7 // Must be odd
#define MEDIAN_SIZE 3 // Must be odd

/* Motion Compensation */

/* Core DSGM */
#define AGE_THRESH  30
#define MEAN_THRESH 2.0 //9.0
#define THETA_D 4 //5
#define VAR_INIT 255 // max unsigned char
#define VAR_MIN 25


DualSGM::DualSGM(){}
DualSGM::~DualSGM(){}

void DualSGM::init(const cv::Mat &first_image)
{
    prev_frame = first_image.clone();
    cv::Size image_size = first_image.size();

    /* Init mem, canidate, apparent */
    bin_mat     = cv::Mat::zeros(image_size, CV_8U);
    app_u_mat   = cv::Mat::zeros(image_size, CV_8U);
//    app_u_mat = first_image.clone();
    app_var_mat = cv::Mat::zeros(image_size, CV_8U);
    app_age_mat = cv::Mat::zeros(image_size, CV_8U);
//    can_u_mat   = cv::Mat::zeros(image_size, CV_8U);
    can_u_mat = first_image.clone();
    can_var_mat = cv::Mat::zeros(image_size, CV_8U);
    can_age_mat = cv::Mat::zeros(image_size, CV_8U);

}


void DualSGM::warpBack(cv::Mat &next_frame, cv::Mat &depth,
                       Eigen::Affine3d &transform,Intrinsic &intrinsic)
{
    warp3D(next_frame,depth,transform,intrinsic);
}


void DualSGM::warpForward(Eigen::Affine3d &transform, Intrinsic &intrinsic)
{
    Eigen::Affine3d forward_transform = transform.inverse();
    warp3D(app_u_mat,prev_depth,forward_transform,intrinsic);
    warp3D(app_var_mat,prev_depth,forward_transform,intrinsic);
    warp3D(can_u_mat,prev_depth,forward_transform,intrinsic);
    warp3D(can_var_mat,prev_depth,forward_transform,intrinsic);
}

void DualSGM::warpForward(const cv::Mat &next_frame, const cv::Mat &depth,
                          Eigen::Affine3d &transform, Intrinsic intrinsic)
{
    warpModel(app_u_mat,next_frame,depth,transform,false,intrinsic);
    warpModel(can_u_mat,next_frame,depth,transform,false,intrinsic);
    warpModel(app_var_mat,next_frame,depth,transform,true,intrinsic);
    warpModel(can_var_mat,next_frame,depth,transform,true,intrinsic);
}


/**
 *  Update model based on next frame and number of tasks, blurring algorithm, motion compesation.
 */

void DualSGM::process(cv::Mat &next_frame, cv::Mat &depth, Eigen::Affine3d &transform,
                      Intrinsic intrinsic, cv::Mat &foreground)
{

    /* Pre processing */
    cv::Mat warped = next_frame.clone();
    cv::Size gb_size = cv::Size(GAUSSIAN_SIZE,GAUSSIAN_SIZE);

    cv::GaussianBlur(warped, warped, gb_size, 0, 0);
    cv::medianBlur(warped, warped, MEDIAN_SIZE);

    if(prev_frame.empty())
    {
        init(warped);
    }
    else{
//        warpBack(warped,depth,transform,intrinsic);
//        warpForward(transform,intrinsic);

        warpForward(warped,depth,transform,intrinsic);
    }


    core_dsgm_update(warped,bin_mat,app_u_mat,app_var_mat,can_u_mat,
                     can_var_mat,app_age_mat,can_age_mat,0, warped.rows);

    prev_frame = next_frame;
    prev_depth = depth;
    foreground = bin_mat;

    //Display output
    cv::imshow("mean", app_u_mat);
    cv::imshow("original", next_frame);

}

void DualSGM::process(const cv::Mat &next_frame, cv::Mat &foreground)
{
    cv::Size gb_size = cv::Size(GAUSSIAN_SIZE,GAUSSIAN_SIZE);

    cv::GaussianBlur(next_frame, next_frame, gb_size, 0, 0);
    cv::medianBlur(next_frame, next_frame, MEDIAN_SIZE);

    if(prev_frame.empty())
    {
        init(next_frame);
    }

    /* Duel Gaussian Model */

    core_dsgm_update(next_frame,bin_mat,app_u_mat,app_var_mat,can_u_mat,
                     can_var_mat,app_age_mat,can_age_mat,0, next_frame.rows);


    prev_frame = next_frame;
    foreground = bin_mat;

    //Display output

    cv::Mat output = cv::Mat(next_frame.rows * 2, next_frame.cols * 2, CV_8U);
    cv::Mat roi = output(cv::Rect(0,0,next_frame.cols,next_frame.rows));
    next_frame.copyTo(roi);
    roi = output(cv::Rect(next_frame.cols,0,next_frame.cols,next_frame.rows));
    foreground.copyTo(roi);
    roi = output(cv::Rect(0,next_frame.rows,next_frame.cols,next_frame.rows));
    app_u_mat.copyTo(roi);
    roi = output(cv::Rect(next_frame.cols,next_frame.rows,next_frame.cols,next_frame.rows));
    can_u_mat.copyTo(roi);
    cv::resize(output,output,cv::Size(next_frame.rows/2, next_frame.cols/2));
    cv::imshow("output",output);

}


/**
 *  Core DSGM update function
 *      next_frame      Next input frame matrix, preprocessed and possibly motion compensated
 *      bin_mat         Output matrix
 *      app_u_mat       Apparent mean matrix
 *      app_var_mat     Apparent variance matrix
 *      can_u_mat       Candidate mean matrix
 *      can_var_mat     Candidate variance matrix
 *      app_ages        Apparent ages matrix
 *      can_ages        Candidate ages matrix
 *      offset          Row offset
 *      row_lim         Row limit
 */
inline void core_dsgm_update(const cv::Mat &next_frame, cv::Mat &bin_mat,
    cv::Mat &app_u_mat, cv::Mat &app_var_mat,
    cv::Mat &can_u_mat, cv::Mat &can_var_mat,
    cv::Mat &app_age_mat, cv::Mat &can_age_mat,
    int offset, int row_lim)
{
    float alpha, V;

    int col_lim = next_frame.cols;

    cv::Mat app_match_mat = cv::Mat::zeros(next_frame.size(), CV_8U);
    cv::Mat can_match_mat = cv::Mat::zeros(next_frame.size(), CV_8U);
    cv::Mat no_match_mat = cv::Mat::zeros(next_frame.size(), CV_8U);
    cv::Mat swapped = cv::Mat::zeros(next_frame.size(), CV_8U);

    /* Core update loop */
    for (int y = offset; y < row_lim; ++y) {
        for (int x = 0; x < col_lim; ++x) {

            uchar intensity = next_frame.at<uchar>(y,x);
            uchar app_u = app_u_mat.at<uchar>(y,x);
            uchar app_var = app_var_mat.at<uchar>(y,x);
            uchar app_age = app_age_mat.at<uchar>(y,x);
            uchar can_u = can_u_mat.at<uchar>(y,x);
            uchar can_var = can_var_mat.at<uchar>(y,x);
            uchar can_age = can_age_mat.at<uchar>(y,x);

            // Get the differences for the candidate and apparent background models
            int adiff = intensity - app_u;
            int cdiff = intensity - can_u;

            if (std::pow(adiff, 2) < MEAN_THRESH * std::max((int)app_var, 25)){

                app_match_mat.at<uchar>(y,x) = std::pow(adiff,2);
                alpha = 1.0 / (double)(app_age + 1);
                app_u = (1.0-alpha) * app_u + alpha * intensity;
                V = std::pow(app_u - intensity,2);
                app_var = (1.0-alpha) * app_var + alpha * V;

                //write into matrix
                app_u_mat.at<uchar>(y,x) = app_u;
                app_var_mat.at<uchar>(y,x) = app_var;

                if (app_age < AGE_THRESH) {
                    app_age_mat.at<uchar>(y,x) ++;
                }

            } else if (pow(cdiff, 2) < MEAN_THRESH * std::max((int)can_var, 25)) {

                can_match_mat.at<uchar>(y,x) = std::pow(cdiff,2);
                alpha = 1.0 / (double)(can_age + 1);
                can_u = (1.0-alpha) * can_u + alpha * intensity;
                V = std::pow(can_u - intensity,2);
                can_var = (1.0-alpha) * can_var + alpha * V;

                //write into matrix
                can_u_mat.at<uchar>(y,x) = can_u;
                can_var_mat.at<uchar>(y,x) = can_var;

                // Cap ages
                if (can_age < AGE_THRESH) {
                    can_age_mat.at<uchar>(y,x) ++;
                }

            } else {

                no_match_mat.at<uchar>(y,x) = intensity;
                can_u_mat.at<uchar>(y, x) = intensity;
                can_var_mat.at<uchar>(y, x) = VAR_INIT;
                can_age_mat.at<uchar>(y,x) = 0;
            }


            app_age = app_age_mat.at<uchar>(y,x);
            can_age = can_age_mat.at<uchar>(y,x);

            if (can_age > app_age) {
                // Swap the models
                swapped.at<uchar>(y,x) = 255;
                app_u_mat.at<uchar>(y,x) = can_u_mat.at<uchar>(y,x);
                app_var_mat.at<uchar>(y,x) = can_var_mat.at<uchar>(y,x);
                app_age_mat.at<uchar>(y,x) = can_age_mat.at<uchar>(y,x);

                can_u_mat.at<uchar>(y, x) = intensity;
                can_var_mat.at<uchar>(y, x) = VAR_INIT;
                can_age_mat.at<uchar>(y,x) = 0;
            }

            int app_diff = app_u_mat.at<uchar>(y,x) - next_frame.at<uchar>(y,x);

            if (std::pow(app_diff, 2) <= THETA_D * std::max(0.25, (double)intensity)) { // 60
                //background
                bin_mat.at<uchar>(y, x) = 0;
            } else {
                //foreground
                bin_mat.at<uchar>(y, x) = 255;
            }

        }
    }

//    cv::imshow("swapped",swapped);
//    cv::imshow("no_mat",no_match_mat);
//    cv::imshow("can_mat",can_match_mat);
//    cv::imshow("app_mat",app_match_mat);

}
