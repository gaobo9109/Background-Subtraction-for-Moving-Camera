#ifndef PARTICLE_FILTER_H
#define PARTICLE_FILTER_H

#include <opencv2/opencv.hpp>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>

#define TRANS_X_STD 0.5
#define TRANS_Y_STD 1.0
#define GAUSS_KERNEL_STD 15.0

typedef struct Particle{
    float x;
    float y;
    float weight;
} Particle;

class ParticleFilter
{
public:

    ParticleFilter(int population, int im_width, int im_height);
    void run(cv::Mat &diffImg, cv::Mat &H);
    void drawParticle(cv::Mat &canvas);

private:
    gsl_rng* rng;
    int nParticle;
    int width_bound,height_bound;
    std::vector<Particle> particles;

    void updateParticle(cv::Mat &H);
    float computeWeight(cv::Mat &diffImg);
    void resampleParticle(float maxWeight);
    float normalizeWeight(float totalWeight);
    bool particleInRange(cv::Point &p);



};

#endif // PARTICLE_FILTER_H
