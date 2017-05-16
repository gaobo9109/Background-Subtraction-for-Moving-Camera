#ifndef MOTION_REMOVAL_H
#define MOTION_REMOVAL_H

#include <opencv2/opencv.hpp>
#include "particle_filter.h"
#include <memory>

class MotionRemoval
{
public:
    MotionRemoval();
    void process(cv::Mat &rgb, cv::Mat &depth);

private:
    std::shared_ptr<ParticleFilter> pf;
    cv::Mat prev_img;


};


#endif // MOTION_REMOVAL_H
