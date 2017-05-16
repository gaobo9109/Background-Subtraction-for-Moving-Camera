#include "particle_filter.h"
#include <math.h>


ParticleFilter::ParticleFilter(int population, int im_width, int im_height)
{
    nParticle = population;
    width_bound = im_width;
    height_bound = im_height;

    //random number generator
    gsl_rng_env_setup();
    rng = gsl_rng_alloc(gsl_rng_mt19937);
    gsl_rng_set(rng, time(NULL));

    for(int i=0; i<nParticle; i++)
    {
        float x = (float)gsl_rng_uniform_int(rng,im_width);
        float y = (float)gsl_rng_uniform_int(rng,im_height);

        Particle p = {x,y,0};
        particles.push_back(p);
    }
}

void ParticleFilter::run(cv::Mat &diffImg, cv::Mat &H)
{
    updateParticle(H);
    float totalWeight = computeWeight(diffImg);
    float maxWeight = normalizeWeight(totalWeight);
    resampleParticle(maxWeight);
}

void ParticleFilter::drawParticle(cv::Mat &canvas)
{
    for(const auto &p : particles)
    {
        cv::Point point(p.x,p.y);
        cv::circle(canvas,point,3,cv::Scalar(0,0,255));
    }
}

void ParticleFilter::updateParticle(cv::Mat &H)
{

    for(auto &p : particles)
    {
        std::vector<cv::Point2f> point, point_warped;
        point.push_back(cv::Point2f(p.x,p.y));
        cv::perspectiveTransform(point,point_warped,H);

        p.x = point_warped[0].x + gsl_ran_gaussian(rng, TRANS_X_STD);
        p.y = point_warped[0].y + gsl_ran_gaussian(rng, TRANS_Y_STD);
    }
}

float ParticleFilter::computeWeight(cv::Mat &diffImg)
{
    assert (diffImg.channels() == 1);

    int totalWeight = 0;
    int i = 0;
    for(auto &p : particles)
    {
        //boundary checking
        if(p.x<0 || p.x>width_bound || p.y<0 || p.y>height_bound)
        {
            p.weight = 0;
            continue;
        }

        float sum = 0;
        float coeff = 1/((std::sqrt(2 * M_PI) * GAUSS_KERNEL_STD));

        //8 neighbours
        sum += diffImg.at<uchar>(p.y,p.x);
        sum += diffImg.at<uchar>(p.y-1,p.x) * coeff * std::exp(-1/(2*std::pow(GAUSS_KERNEL_STD,2)));
        sum += diffImg.at<uchar>(p.y+1,p.x) * coeff * std::exp(-1/(2*std::pow(GAUSS_KERNEL_STD,2)));
        sum += diffImg.at<uchar>(p.y,p.x-1) * coeff * std::exp(-1/(2*std::pow(GAUSS_KERNEL_STD,2)));
        sum += diffImg.at<uchar>(p.y,p.x+1) * coeff * std::exp(-1/(2*std::pow(GAUSS_KERNEL_STD,2)));
        sum += diffImg.at<uchar>(p.y-1,p.x-1) * coeff * std::exp(-1/std::pow(GAUSS_KERNEL_STD,2));
        sum += diffImg.at<uchar>(p.y-1,p.x+1) * coeff * std::exp(-1/std::pow(GAUSS_KERNEL_STD,2));
        sum += diffImg.at<uchar>(p.y+1,p.x+1) * coeff * std::exp(-1/std::pow(GAUSS_KERNEL_STD,2));
        sum += diffImg.at<uchar>(p.y+1,p.x-1) * coeff * std::exp(-1/std::pow(GAUSS_KERNEL_STD,2));

        p.weight = sum;
        totalWeight += sum;

    }
    return totalWeight;
}

void ParticleFilter::resampleParticle(float maxWeight)
{
    //resampling wheel
    std::vector<Particle> newParticles;
    float beta = 0;
    int index = gsl_rng_get(rng) % nParticle;

    for(int i=0; i<nParticle; i++)
    {
        beta += (float)gsl_rng_uniform(rng) * 2 * maxWeight;
        while(beta > particles[index].weight)
        {
            beta -= particles[index].weight;
            index = (index + 1) % nParticle;
        }

        //copy the particle
        Particle p = {particles[index].x,particles[index].y,particles[index].weight};
        newParticles.push_back(p);
    }

    particles = newParticles;

}

float ParticleFilter::normalizeWeight(float totalWeight)
{
    float maxWeight = 0;
    for(auto &p : particles)
    {
        p.weight /= totalWeight;
        maxWeight = std::max(maxWeight,p.weight);
    }

    return maxWeight;
}



