#include "dual_sgm.h"


void read_video_files(std::string video_file)
{
    cv::VideoCapture capture(video_file);
    DualSGM sgm;
    cv::Mat img,foreground,frame;

    while(capture.isOpened())
    {
        capture >> img;
        if(img.empty())break;

        cv::cvtColor(img,frame,CV_BGR2GRAY);

        sgm.process(frame,foreground);

        cv::imshow("original",img);
        cv::imshow("foreground",foreground);
        if(cv::waitKey(30) >= 0) break;
    }
}

void read_image_files(cv::String image_folder)
{
    std::vector<cv::String> filenames;
    cv::glob(image_folder,filenames);

    DualSGM sgm;
    cv::Mat img,foreground,frame;
    int key;

    int count = 0;

    for(int i=0; i<filenames.size(); i++)
    {
        img = cv::imread(filenames[i]);
        cv::cvtColor(img,frame,CV_BGR2GRAY);

        sgm.process(frame,foreground);

        cv::imshow("original",img);
        cv::imshow("foreground",foreground);
        cv::waitKey(30);
    }
}

int main(int argc,char **argv)
{
	std::string file = argv[1];
	read_video_files(file);
	// read_image_files("../data/images");

}
