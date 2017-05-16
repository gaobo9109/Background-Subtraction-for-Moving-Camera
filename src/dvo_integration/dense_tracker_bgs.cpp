#include <dvo/dense_tracking.h>
#include <dvo/core/intrinsic_matrix.h>
#include <dvo_ros/CameraDenseTrackerConfig.h>
#include <dvo_ros/camera_base.h>
#include <ros/ros.h>
#include <ros/console.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <image_transport/image_transport.h>
#include <boost/shared_ptr.hpp>
#include <bgs_util/image_transformer.h>
#include <bgs_util/surface_pyramid.h>
#include "../gmm/gmm_zikovic.h"
#include "../gmm/gmm_bowden.h"
#include "../gmm/gmm_with_depth.h"

using namespace dvo_ros;
using namespace dvo::core;
using namespace dvo;
using namespace GMM_WITH_DEPTH;
//using namespace GMM_BOWDEN;
//using namespace GMM_ZIKOVIC;

class DenseTrackerBGS : public CameraBase
{
private:
    uint32_t width;
    uint32_t height;
    boost::shared_ptr<BGModelMOG> subtractor;
    boost::shared_ptr<DenseTracker> tracker;
    DenseTracker::Config tracker_cfg;
    RgbdCameraPyramidPtr camera;
    RgbdImagePyramidPtr current, reference;

    image_transport::Publisher fg_img_pub;
    image_transport::Publisher bg_img_pub;

    bool hasChanged(const sensor_msgs::CameraInfo::ConstPtr& camera_info_msg);
    void reset(const sensor_msgs::CameraInfo::ConstPtr& camera_info_msg);
    void publishImgMsg(std_msgs::Header &header, const std::string &enconding, cv::Mat &img, image_transport::Publisher &pub);

public:
    DenseTrackerBGS(ros::NodeHandle& nh, ros::NodeHandle& nh_private);
    virtual void handleImages(
            const sensor_msgs::Image::ConstPtr& rgb_image_msg,
            const sensor_msgs::Image::ConstPtr& depth_image_msg,
            const sensor_msgs::CameraInfo::ConstPtr& rgb_camera_info_msg,
            const sensor_msgs::CameraInfo::ConstPtr& depth_camera_info_msg);

};

DenseTrackerBGS::DenseTrackerBGS(ros::NodeHandle& nh, ros::NodeHandle& nh_private):
    CameraBase(nh, nh_private)
{
    startSynchronizedImageStream();
    image_transport::ImageTransport it(nh);
    fg_img_pub = it.advertise("foreground", 1);
    bg_img_pub = it.advertise("background", 1);
    width = 0;
    height = 0;
}

bool DenseTrackerBGS::hasChanged(const sensor_msgs::CameraInfo::ConstPtr &camera_info_msg)
{
    return width != camera_info_msg->width || height != camera_info_msg->height;
}

void DenseTrackerBGS::reset(const sensor_msgs::CameraInfo::ConstPtr &camera_info_msg)
{
    IntrinsicMatrix intrinsics = IntrinsicMatrix::create(camera_info_msg->P[0],camera_info_msg->P[5], camera_info_msg->P[2], camera_info_msg->P[6]);
    Intrinsic intrinsics1 = {camera_info_msg->P[0],camera_info_msg->P[5], camera_info_msg->P[2], camera_info_msg->P[6]};

    camera.reset(new RgbdCameraPyramid(camera_info_msg->width, camera_info_msg->height, intrinsics));
    camera->build(tracker_cfg.getNumLevels());

    tracker.reset(new DenseTracker(tracker_cfg));
    subtractor.reset(new BGModelMOG);

    static RgbdImagePyramid* const __null__ = 0;

    reference.reset(__null__);
    current.reset(__null__);

    width = camera_info_msg->width;
    height = camera_info_msg->height;
}

void DenseTrackerBGS::handleImages(
        const sensor_msgs::Image::ConstPtr& rgb_image_msg,
        const sensor_msgs::Image::ConstPtr& depth_image_msg,
        const sensor_msgs::CameraInfo::ConstPtr& rgb_camera_info_msg,
        const sensor_msgs::CameraInfo::ConstPtr& depth_camera_info_msg)
{
    if(hasChanged(rgb_camera_info_msg))
    {
        ROS_WARN("RGB image size has changed, resetting tracker!");
        reset(rgb_camera_info_msg);
    }

//    std::cout << "received" << std::endl;
    cv::Mat intensity, depth;
    cv::Mat rgb_in = cv_bridge::toCvShare(rgb_image_msg)->image;

    if(rgb_in.channels() == 3)
    {
        cv::Mat tmp;
        cv::cvtColor(rgb_in, tmp, CV_BGR2GRAY, 1);

        tmp.convertTo(intensity, CV_32F);
    }
    else
    {
        rgb_in.convertTo(intensity, CV_32F);
    }

    cv::Mat depth_in = cv_bridge::toCvShare(depth_image_msg)->image;

    if(depth_in.type() == CV_16UC1)
    {
        SurfacePyramid::convertRawDepthImageSse(depth_in, depth, 0.001);
    }
    else
    {
        depth = depth_in;
    }

    reference.swap(current);
    current = camera->create(intensity, depth);

    if(!reference) return;

    Eigen::Affine3d transform, reverse_transform;
    bool success = tracker->match(*reference, *current, transform);
    reverse_transform = transform.inverse();

    cv::Mat foreground,background;
    bool isIdentity = transform.matrix().isIdentity(1e-3);
//    if(isIdentity)
//        subtractor->process(rgb_in,foreground,background);
    subtractor->process(rgb_in,depth,foreground,background);
//    else
//        subtractor->process(rgb_in,depth,transform,foreground,background,0.25);

    //convert back to ros message
    std_msgs::Header header;
    header.stamp = ros::Time::now();
    publishImgMsg(header,"mono8",foreground,fg_img_pub);
    publishImgMsg(header,"bgr8",background,bg_img_pub);
}

void DenseTrackerBGS::publishImgMsg(std_msgs::Header &header, const std::string &encoding,
                                    cv::Mat &img, image_transport::Publisher &pub)
{
    sensor_msgs::ImagePtr img_msg;
    img_msg = cv_bridge::CvImage(header,encoding,img).toImageMsg();

    if(pub.getNumSubscribers() > 0)
        pub.publish(img_msg);
}



int main(int argc, char **argv) {
    ros::init(argc, argv, "tracker");

    ros::NodeHandle nh;
    ros::NodeHandle nh_private("~");
    DenseTrackerBGS tracker(nh, nh_private);
    std::cout << "node started" << std::endl;
    ros::MultiThreadedSpinner spinner(4);
    spinner.spin();

    return 0;
}
