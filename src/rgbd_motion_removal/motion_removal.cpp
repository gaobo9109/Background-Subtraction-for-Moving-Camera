#include "motion_removal.h"
#include "../util/warp_image.h"
#include "../util/surface_pyramid.h"

MotionRemoval::MotionRemoval()
{

}

void MotionRemoval::initPF(int nParticle, int im_width, int im_height)
{
    pf = std::shared_ptr<ParticleFilter>(new ParticleFilter(nParticle,im_width,im_height));
}


void MotionRemoval::process(cv::Mat &rgb, cv::Mat &depth)
{
    if(prev_img.empty())
    {
        prev_img = rgb;
        initPF(300, rgb.cols,rgb.rows);
        cv::Mat canvas = rgb.clone();
        pf->drawParticle(canvas);
        cv::imshow("particle",canvas);
        return;
    }

    cv::Mat warped, diffImg;
//    cv::Mat H = warp2D(warped,prev_img,rgb);
//    cv::absdiff(warped,rgb,diffImg);
//    cv::imshow("diff",diffImg);
//    pf->run(diffImg,H);
//    cv::Mat canvas = rgb.clone();
//    pf->drawParticle(canvas);
//    cv::imshow("particle",canvas);
//    prev_img = rgb;
}


