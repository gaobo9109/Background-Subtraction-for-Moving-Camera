#include <dvo/dense_tracking.h>
#include <dvo/core/intrinsic_matrix.h>
#include <dvo_ros/CameraDenseTrackerConfig.h>
#include <boost/shared_ptr.hpp>
#include <bgs_util/image_transformer.h>
#include <bgs_util/surface_pyramid.h>
#include <bgs_util/file_reader.h>

using namespace dvo::core;
using namespace dvo;

class DVOTest
{
private:
    uint32_t width;
    uint32_t height;

    boost::shared_ptr<dvo::DenseTracker> tracker;
    boost::shared_ptr<ImageTransformer> transformer;
    DenseTracker::Config tracker_cfg;
    RgbdCameraPyramidPtr camera;
    RgbdImagePyramidPtr current, reference;

    cv::Mat prev_img;

public:
    DVOTest(Intrinsic intrinsic);
    void handleImages(cv::Mat &rgb_in, cv::Mat &depth_in);
};

DVOTest::DVOTest(Intrinsic intrinsic)
{
    tracker_cfg = DenseTracker::getDefaultConfig();
    IntrinsicMatrix intrinsicMat = IntrinsicMatrix::create(intrinsic.fx, intrinsic.fy, intrinsic.ox, intrinsic.oy);
    camera.reset(new dvo::core::RgbdCameraPyramid(640, 480, intrinsicMat));
    camera->build(tracker_cfg.getNumLevels());

    tracker.reset(new DenseTracker(tracker_cfg));
    transformer.reset(new ImageTransformer(intrinsic));
    transformer->setImageDimension(640,480);

    static RgbdImagePyramid* const __null__ = 0;

    reference.reset(__null__);
    current.reset(__null__);
}

void DVOTest::handleImages(cv::Mat &rgb_in, cv::Mat &depth_in)
{
    Eigen::Affine3d transform;
    cv::Mat intensity, depth;

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

    if(depth_in.type() == CV_16UC1)
    {
        SurfacePyramid::convertRawDepthImageSse(depth_in, depth, 0.0002);
    }
    else
    {
        depth = depth_in;
    }

    reference.swap(current);
    current = camera->create(intensity, depth);

    if(!reference)
    {
        prev_img = rgb_in;
        return;
    }

    bool success = tracker->match(*reference, *current, transform);
    cv::Mat model = prev_img.clone();
    cv::Mat model1 = prev_img.clone();
    transformer->warp3DRChannelSwap(model,rgb_in,depth,transform);
    transformer->warp3D(model1,rgb_in,depth,transform);
    cv::imshow("img",model);
    static int count = 0;
    std::string name = boost::str(boost::format("%sexp2_%d.jpg") % "/home/gaobo/test_result/25_04_2017/" % count++) ;
//    cv::imwrite(name,model);
//    prev_img = rgb_in;
    prev_img = model1;
}

int main()
{
    Intrinsic intrinsic = {520.9, 521.0, 325.1, 249.7};
    DVOTest dvo(intrinsic);
    FileReader reader("/home/gaobo/dataset/fr2_desk/");
    std::vector<std::string> rgb_list, depth_list;
    reader.readRGBD(rgb_list,depth_list);

    for(int i=0; i<rgb_list.size(); i++)
    {
        cv::Mat rgb = cv::imread(rgb_list[i]);
        cv::Mat depth = cv::imread(depth_list[i]);
        dvo.handleImages(rgb,depth);
        if(i != 0)
            while(cv::waitKey(0) != 'c');
    }

}
