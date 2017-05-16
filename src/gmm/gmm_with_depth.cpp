#include "gmm_with_depth.h"

namespace GMM_WITH_DEPTH
{
BGModelMOG::BGModelMOG()
{
    frameSize = Size(0,0);
    frameType = 0;

    nframes = 0;
    nmixtures = defaultNMixtures;
    history = defaultHistory;
    varThreshold = defaultVarThreshold;
    backgroundRatio = defaultBackgroundRatio;
    initialVar = defaultVarInit;
    initialWeight = defaultInitialWeight;
}

void BGModelMOG::initialize(Size _frameSize, int _frameType)
{
    frameSize = _frameSize;
    frameType = _frameType;
    nframes = 0;

    int nchannels = CV_MAT_CN(frameType);

    bgModelColor.create(1, frameSize.height * frameSize.width * nmixtures * 8, CV_32F);
    bgModelDepth.create(1, frameSize.height * frameSize.width * nmixtures * 4, CV_32F);

    bgModelColor = Scalar::all(0);
    bgModelDepth = Scalar::all(0);
}

void BGModelMOG::updateModel(Mat &image, Mat &depth, Mat &fgmask, Mat &bgimg, float learningRate)
{
    //initialize output mask and background image
    fgmask.create(frameSize, CV_8U);
//    bgimg.create(frameSize,CV_8U);
    bgimg.create(frameSize,frameType);
    bgimg = Scalar::all(0);
    fgmask = Scalar::all(0);

    Mat transformed;
    cvtColor(image,transformed,CV_BGR2YCrCb);
    const uchar* src_color = transformed.ptr<uchar>();
    const float* src_depth = depth.ptr<float>();
    uchar* dst = fgmask.ptr<uchar>();
    uchar* bgmean = bgimg.ptr<uchar>();
    ColorModel* cm = (ColorModel*)bgModelColor.data;
    DepthModel* dm = (DepthModel*)bgModelDepth.data;

    float alpha = learningRate;

    for(int y=0; y<image.rows; y++)
    {
        for(int x=0; x<image.cols; x++, src_color+=3, src_depth++, cm+=nmixtures, dm+=nmixtures, dst++, bgmean+=3)
        {
            Vec3f colorVal(src_color[0], src_color[1], src_color[2]);
            float depthVal = *src_depth;

            // process depth models
            bool fgDepth = processDepth(dm,depthVal,alpha);

            // process color models
            bool fgColor = processColor(cm,colorVal,fgDepth,alpha);

            dst[0] = fgColor ? 255 : 0;
            bgmean[0] = cm[0].mean[0];
            bgmean[1] = cm[0].mean[1];
            bgmean[2] = cm[0].mean[2];
        }
    }
    cvtColor(bgimg,bgimg,CV_YCrCb2BGR);
    medianBlur(fgmask,fgmask,5);
}

bool BGModelMOG::processDepth(DepthModel *dm, float currVal, float alpha)
{
    bool fgDepth = false;
    int kHit = -1, kForeground = -1;
    float wsum = 0;
    float initialVarDepth = 1.0f;
    float varThresholdDepth = 1.5*1.5;
    int k;

    if(!std::isnan(currVal))
    {
        for(k = 0; k < nmixtures; k++)
        {
            float w = dm[k].weight;
            wsum += w;
            if( w < FLT_EPSILON )
                break;
            float mu = dm[k].mean;
            float var = dm[k].var;
            float d2 = currVal - mu;
            if(d2*d2 < varThresholdDepth * var)
            {
                wsum -= w;
                float dw = alpha * (1.f - w);
                dm[k].weight += dw;
                dm[k].mean += alpha * d2;
                dm[k].var = std::max(var + alpha * (d2 * d2 - var), initialVarDepth); //another intial var value?
                dm[k].sortKey = w / std::sqrt(var);

                for(int k1 = k-1; k1 >= 0; k1-- )
                {
                    if(dm[k1].sortKey >= dm[k1+1].sortKey )
                        break;
                    std::swap(dm[k1], dm[k1+1] );
                }

                kHit = k;
                break;
            }
        }

        if( kHit < 0 )
        {
            kHit = k = std::min(k, nmixtures-1);
            wsum += initialWeight - dm[k].weight;
            dm[k].weight = alpha;
            dm[k].mean = currVal;
            dm[k].var = initialVarDepth;
            dm[k].sortKey = initialWeight / std::sqrt(initialVarDepth);
        }
        else
            for( ; k < nmixtures; k++ )
                wsum += dm[k].weight;

        float wscale = 1.f/wsum;
        wsum = 0;
        for( k = 0; k < nmixtures; k++ )
        {
            wsum += dm[k].weight *= wscale;
            dm[k].sortKey *= wscale;
            if( wsum > backgroundRatio && kForeground < 0 )
                kForeground = k+1;
        }

        fgDepth = kHit >= kForeground;
    }

    return fgDepth;
}

bool BGModelMOG::processColor(ColorModel *cm, Vec3f currVal, bool fgDepth, float alpha)
{
    bool fgColor = false;
    int kHit = -1, kForeground = -1;
    float wsum = 0;
    int k;

    for(k = 0; k < nmixtures; k++)
    {
        float w = cm[k].weight;
        wsum += w;
        if( w < FLT_EPSILON )
            break;
        Vec3f mu = cm[k].mean;
        Vec3f var = cm[k].var;
        Vec3f diff = currVal - mu;
        float d2 = diff.dot(diff);
        if(d2 < varThreshold * var[0] + var[1] + var[2])
        {
            wsum -= w;
            float dw = alpha*(1.f - w);
            cm[k].weight += dw;
            cm[k].mean += alpha * diff;
            var = Vec3f(std::max(var[0] + alpha*(diff[0]*diff[0] - var[0]), initialVar),
                    std::max(var[1] + alpha*(diff[1]*diff[1] - var[1]), initialVar),
                    std::max(var[2] + alpha*(diff[2]*diff[2] - var[2]), initialVar));
            cm[k].var = var;
            cm[k].sortKey = w/std::sqrt(var[0] + var[1] + var[2]);

            for(int k1 = k-1; k1 >= 0; k1-- )
            {
                if(cm[k1].sortKey >= cm[k1+1].sortKey )
                    break;
                std::swap(cm[k1], cm[k1+1] );
            }

            kHit = k;
            break;
        }
    }

    if( kHit < 0 )
    {
        kHit = k = std::min(k, nmixtures-1);
        wsum += initialWeight - cm[k].weight;
        cm[k].weight = initialWeight;
        cm[k].mean = currVal;
        cm[k].var = Vec3f(initialVar,initialVar,initialVar);
        cm[k].sortKey = initialWeight / std::sqrt(3 * initialVar);
    }
    else
        for( ; k < nmixtures; k++ )
            wsum += cm[k].weight;

    float wscale = 1.f/wsum;
    wsum = 0;
    for( k = 0; k < nmixtures; k++ )
    {
        wsum += cm[k].weight *= wscale;
        cm[k].sortKey *= wscale;
        if( wsum > backgroundRatio && kForeground < 0 )
            kForeground = k+1;
    }

    fgColor = kHit >= kForeground;
    return fgColor;
}

void BGModelMOG::process(Mat &image, Mat &depth, Mat &fgmask, Mat &background, float learningRate)
{
    if(prev_img.empty()){
        initialize(image.size(), image.type());
//        std::cout << depth << std::endl;
    }

    ++nframes;
    learningRate = learningRate >= 0 && nframes > 1 ? learningRate : 1./std::min( 2*nframes, history);
    updateModel(image,depth,fgmask,background,learningRate);

    prev_img = image;
}

void BGModelMOG::process(Mat &image, Mat &depth, Eigen::Affine3d &transform, Mat &fgmask, Mat &background, float learningRate)
{

}

} //namespace GMM_WITH_DEPTH
