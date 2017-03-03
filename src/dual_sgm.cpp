#include "dual_sgm.h"
#include <cmath>

Model::Model()
{
    curr_mean = 0;
    curr_var = 0;
    curr_age = 0;

    can_mean = 0;
    can_var = 0;
    can_age = 0;

    update_index = 0;
}

void Model::update_mean(float pixel_mean)
{
    float learning_rate;
    if(std::pow(pixel_mean-curr_mean, 2) < VAR_THRESH_MODEL_MATCH * curr_var)
    {
        learning_rate = 1.0 / (curr_age + 1);
        update_index = 0;
        curr_mean = (1-learning_rate) * curr_mean + learning_rate * pixel_mean;
    }

    else if(std::pow(pixel_mean-can_mean,2) < VAR_THRESH_MODEL_MATCH * can_var)
    {
        learning_rate = 1.0 / (can_age + 1);
        update_index = 1;
        can_mean = (1-learning_rate) * can_mean + learning_rate * pixel_mean;
    }

    else
    {
        can_mean = pixel_mean;
        can_age = 0;
        update_index = 1;
    }
}

void Model::update_var_age(float pixel_var)
{

    float learning_rate;
    if(update_index==0)
    {
        learning_rate = 1 / (curr_age + 1);
        curr_var = std::max((1-learning_rate) * curr_var + learning_rate * pixel_var, MIN_BG_VAR);
        curr_age = std::min(curr_age + 1,MAX_BG_AGE);
    }
    else
    {
        learning_rate = 1 / (can_age + 1);
        can_var = std::max((1-learning_rate) * can_var + learning_rate * pixel_var, MIN_BG_VAR);
        can_age = std::min(can_age + 1, MAX_BG_AGE);
    }
}

void Model::swap_model(float pixel_mean)
{
    if(can_age > curr_age)
    {
        curr_mean = can_mean;
        curr_var = can_var;
        curr_age = can_age;

        can_mean = pixel_mean;
        can_var = INIT_BG_VAR;
        can_age = 0;
    }
}

void Model::mix_models(std::vector<Model> &models, std::vector<float> &weights, bool is_edge_model)
{

    if(models.size()==0) return;

    float weight_sum = 0;
    float curr_mean_mixed = 0;
    float curr_var_mixed = 0;
    float curr_age_mixed = 0;

    float can_mean_mixed = 0;
    float can_var_mixed = 0;
    float can_age_mixed = 0;

    //calculate weighted mean and weighted age
    for(int i=0; i<models.size(); i++)
    {
        weight_sum += weights[i];
        curr_mean_mixed += weights[i] * models[i].curr_mean;
        curr_age_mixed += weights[i] * models[i].curr_age;

        can_mean_mixed += weights[i] * models[i].can_mean;
        can_age_mixed += weights[i] * models[i].can_age;
    }

    curr_mean_mixed /= weight_sum;
    curr_age_mixed /= weight_sum;
    can_mean_mixed /= weight_sum;
    can_age_mixed /= weight_sum;

    //calculate weighted variance
    for(int i=0; i<models.size(); i++)
    {
        curr_var_mixed += weights[i] * (models[i].curr_var + VARIANCE_INTERPOLATE_PARAM *
                          std::pow(curr_mean_mixed-models[i].curr_mean,2));

        can_var_mixed += weights[i] * (models[i].can_var + VARIANCE_INTERPOLATE_PARAM *
                         std::pow(can_mean_mixed-models[i].can_mean,2));
    }

    curr_var_mixed /= weight_sum;
    can_var_mixed /= weight_sum;

    //adjustment
    curr_mean = curr_mean_mixed;
    curr_var = std::max(curr_var_mixed, MIN_BG_VAR);
    can_mean = can_mean_mixed;
    can_var = std::max(can_var_mixed,MIN_BG_VAR);

    if(is_edge_model)
    {
        curr_var = INIT_BG_VAR;
        curr_age = 0;
        can_var = INIT_BG_VAR;
        can_age = 0;
    }

    else
    {
        curr_age = std::min(curr_age_mixed * std::exp(-VAR_DEC_RATIO *
                   std::max(0.0f,curr_var-VAR_MIN_NOISE_T)),(double)MAX_BG_AGE);

        can_age = std::min(can_age_mixed * std::exp(-VAR_DEC_RATIO *
                  std::max(0.0f,can_var-VAR_MIN_NOISE_T)),(double)MAX_BG_AGE);

    }

}


/*
 * DUAL SGM
*/

void DualSGM::init_model(cv::Mat &img)
{
    img_height = img.rows;
    img_width = img.cols;

    model_height = img_height/BLOCK_SIZE;
    model_width = img_width/BLOCK_SIZE;
    model_size = model_width * model_height;

    bg_model = std::vector<Model>(model_size,Model());
}

void DualSGM::update_mean(cv::Mat &img)
{
    for(int i=0; i<bg_model.size(); i++)
    {
        int sum = 0;
        int base_row = i/model_width;
        int base_col = i%model_width;

        //find average intensity in a grid
        for(int ii=0; ii<BLOCK_SIZE; ii++)
        {
            for(int jj=0; jj<BLOCK_SIZE; jj++)
            {
                int row = base_row * BLOCK_SIZE + ii;
                int col = base_col * BLOCK_SIZE + jj;

                sum += img.at<uchar>(row,col);
            }
        }

        float pixel_mean = sum / BLOCK_SIZE_SQR;


        bg_model[i].swap_model(pixel_mean);
        bg_model[i].update_mean(pixel_mean);
    }
}

void DualSGM::update_var_age(cv::Mat &img)
{
    for(int i=0; i<bg_model.size(); i++)
    {
        float mean = bg_model[i].get_mean();
        float max_var = 0;
        int base_row = i/model_width;
        int base_col = i%model_width;

        //find average intensity in a grid
        for(int ii=0; ii<BLOCK_SIZE; ii++)
        {
            for(int jj=0; jj<BLOCK_SIZE; jj++)
            {
                int row = base_row * BLOCK_SIZE + ii;
                int col = base_col * BLOCK_SIZE + jj;

                float var = std::pow(mean - img.at<uchar>(row,col), 2);
                if(var > max_var) max_var = var;
            }
        }

        bg_model[i].update_var_age(max_var);
    }
}

void DualSGM::get_foreground_map(cv::Mat &img, cv::Mat &output)
{
    output.create(img.size(),CV_8U);

    for(int i=0; i<model_size; i++)
        {
            int base_row = i/model_width;
            int base_col = i%model_width;

            for(int ii=0; ii<BLOCK_SIZE; ii++)
            {
                for(int jj=0; jj<BLOCK_SIZE; jj++)
                {
                    int row = base_row * BLOCK_SIZE + ii;
                    int col = base_col * BLOCK_SIZE + jj;

                    float intensity = (float)img.at<uchar>(row,col);
                    if(bg_model[i].is_foreground_pixel(intensity))
                    {
                        output.at<uchar>(row,col) = 255;
                    }
                    else
                    {
                        output.at<uchar>(row,col) = 0;
                    }
                }
            }

        }
}

cv::Mat DualSGM::KLT(cv::Mat &img)
{
    assert(!prev_img.empty());

    std::vector<cv::Point2f>   prevPts;
    std::vector<cv::Point2f>   nextPts;
    std::vector<unsigned char> status;
    std::vector<float>         error;

    int maxCorners = 100;
    double qualityLevel = 0.01;
    double minDistance = 5;
    cv::Mat mask;
    int blockSize = 7;

    cv::goodFeaturesToTrack(prev_img,prevPts,maxCorners,qualityLevel,minDistance,mask,blockSize);
    cv::calcOpticalFlowPyrLK(prev_img,img,prevPts,nextPts,status,error);

    std::vector<cv::Point2f> prevPts2;
    std::vector<cv::Point2f> nextPts2;

    for(int i=0; i<status.size(); i++)
    {
        if(status[i])
        {
            prevPts2.push_back(prevPts[i]);
            nextPts2.push_back(nextPts[i]);
        }
    }

    cv::Mat warped;
    cv::Mat H = cv::findHomography(prevPts2,nextPts2,mask,CV_RANSAC);

    cv::warpPerspective(img,warped,H,img.size(),CV_INTER_LINEAR | CV_WARP_INVERSE_MAP);
    cv::Mat residue,residue1;
    cv::absdiff(prev_img,warped,residue);
    cv::absdiff(prev_img,img,residue1);

    cv::imshow("previous", prev_img);
    cv::imshow("warped",warped);
    cv::imshow("residue", residue);
    cv::imshow("residue1",residue1);

    return H.inv();
}


void DualSGM::motion_compensate(cv::Mat &H)
{
    for(int i=0; i<bg_model.size(); i++)
    {
        int row = i/model_width;
        int col = i%model_width;

        float x = BLOCK_SIZE * col + BLOCK_SIZE / 2 - 0.5;
        float y = BLOCK_SIZE * row + BLOCK_SIZE / 2 - 0.5;

        std::vector<cv::Point2f> vec;
        vec.push_back(cv::Point2f(x,y));
        cv::perspectiveTransform(vec,vec,H);

        float warped_i = vec.at(0).y / BLOCK_SIZE;
        float warped_j = vec.at(0).x / BLOCK_SIZE;


        int idx_i = std::floor(warped_i + 0.5);
        int idx_j = std::floor(warped_j + 0.5);

        float di = warped_i - (float)idx_i;
        float dj = warped_j - (float)idx_j;

//        int idx_i = std::floor(warped_i);
//        int idx_j = std::floor(warped_j);

//        float di = warped_i - ((float)idx_i + 0.5);
//        float dj = warped_j - ((float)idx_j + 0.5);


        float w_h = std::abs(dj) * (1.0 - std::abs(di));
        float w_v = std::abs(di) * (1.0 - std::abs(dj));
        float w_hv = std::abs(di) * std::abs(dj);
        float w_self = (1.0 - std::abs(di)) * (1.0 - std::abs(dj));

        std::vector<Model> models;
        std::vector<float> weights;
        int model_idx;
        int adjusted_i,adjusted_j;

        //w_h
        if(dj != 0)
        {
            adjusted_j = idx_j + (int)copysign(1.0, dj);
            if(is_valid_index(idx_i,adjusted_j))
            {
                model_idx = idx_i * model_width + adjusted_j;
                models.push_back(bg_model[model_idx]);
                weights.push_back(w_h);
            }
        }

        //w_v
        if(di != 0)
        {
            adjusted_i = idx_i + (int)copysign(1.0, di);
            if(is_valid_index(adjusted_i,idx_j))
            {
                model_idx = adjusted_i * model_width + idx_j;
                models.push_back(bg_model[model_idx]);
                weights.push_back(w_v);
            }
        }

        //w_hv
        if(di != 0 && dj != 0)
        {
            adjusted_i = idx_i + (int)copysign(1.0, di);
            adjusted_j = idx_j + (int)copysign(1.0, dj);
            if(is_valid_index(adjusted_i,adjusted_j))
            {
                model_idx = adjusted_i * model_width + adjusted_j;
                models.push_back(bg_model[model_idx]);
                weights.push_back(w_hv);
            }
        }

        //w_self
        if(is_valid_index(idx_i,idx_j))
        {
            model_idx = idx_i * model_width + idx_j;
            models.push_back(bg_model[model_idx]);
            weights.push_back(w_self);
        }

        bool is_edge_model = idx_i < 1 || idx_i >= model_height-1 || idx_j < 1 || idx_j>= model_width-1;

        bg_model[i].mix_models(models,weights, is_edge_model);
    }
}



void DualSGM::process(cv::Mat &img, cv::Mat &foreground)
{
    cv::GaussianBlur(img, img, cv::Size(5,5), 0, 0);
    cv::medianBlur(img,img, 3);

    if(prev_img.empty()){
        init_model(img);
    }
    // else
    // {
    //     cv::Mat H = KLT(img);
    //     motion_compensate(H);
    // }
    prev_img = img.clone();

    update_mean(img);
    update_var_age(img);
    get_foreground_map(img,foreground);
}


