#ifndef DUAL_SGM_H_
#define DUAL_SGM_H_

#include "param.h"
#include <opencv2/opencv.hpp>

class Model
{
public:
    float curr_mean;
    float curr_var;
    int curr_age;

    float can_mean;
    float can_var;
    int can_age;

    int update_index;

    Model();
    ~Model(){}

    void update_mean(float pixel_mean);
    void update_var_age(float pixel_var);
    void swap_model(float pixel_mean);
    void mix_models(std::vector<Model> &models, std::vector<float> &weights, bool is_edge_model);

    inline float get_mean(){ return update_index == 0 ? curr_mean : can_mean; }
    inline bool is_foreground_pixel(float intensity)
    {
        return std::pow(intensity-curr_mean,2) > VAR_THRESH_FG_DETERMINE * curr_var;
    }
};


class DualSGM
{
private:

    int model_width,model_height,model_size;
    int img_width,img_height;
    std::vector<Model> bg_model;
    cv::Mat prev_img;

    void update_mean(cv::Mat &img);
    void update_var_age(cv::Mat &img);
    cv::Mat KLT(cv::Mat &img);
    void motion_compensate(cv::Mat &H);
    void get_foreground_map(cv::Mat &img, cv::Mat &output);


    inline bool is_valid_index(int row, int col)
    {
        return row >= 0 && row < model_height && col >= 0 && col < model_width;
    }

public:
    DualSGM(){}
    ~DualSGM(){}

    void init_model(cv::Mat &img);
    void process(cv::Mat &img, cv::Mat &foreground);

};




#endif /* DUAL_SGM_H_ */
