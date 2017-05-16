#include "gmm_bowden.h"

namespace GMM_BOWDEN{

BGModelMOG::BGModelMOG()
{
    frameSize = Size(0,0);
    frameType = 0;

    nframes = 0;
    nmixtures = defaultNMixtures;
    history = defaultHistory;
    varThreshold = defaultVarThreshold;
    backgroundRatio = defaultBackgroundRatio;
    noiseSigma = defaultNoiseSigma;
    initialWeight = defaultInitialWeight;
}

void BGModelMOG::initialize(Size _frameSize, int _frameType)
{
    frameSize = _frameSize;
    frameType = _frameType;
    nframes = 0;

    int nchannels = CV_MAT_CN(frameType);

    bgmodel.create( 1, frameSize.height*frameSize.width*nmixtures*(2 + 2*nchannels), CV_32F );
    bgmodel = Scalar::all(0);
}

void BGModelMOG::updateModel(Mat &image, Mat &fgmask, Mat &bgimg, float learningRate)
{
    int x, y, k, k1, rows = image.rows, cols = image.cols;
    float alpha = learningRate, T = backgroundRatio, vT = varThreshold;
    int K = nmixtures;

    const float w0 = initialWeight;
    const float sk0 = w0/(noiseSigma * 2 * std::sqrt(3.));
    const float var0 = noiseSigma * noiseSigma*4;
    const float minVar = noiseSigma * noiseSigma;
    MixData* mptr = (MixData*)bgmodel.data;
    const uchar* src = image.ptr<uchar>();
    uchar* dst = fgmask.ptr<uchar>();
    uchar* bgmean = bgimg.ptr<uchar>();

    for( y = 0; y < rows; y++ )
    {
        for( x = 0; x < cols; x++, mptr += K, src += 3, dst ++, bgmean += 3)
        {
            float wsum = 0;
            Vec3f pix(src[0], src[1], src[2]);
            int kHit = -1, kForeground = -1;

            for( k = 0; k < K; k++ )
            {
                float w = mptr[k].weight;
                wsum += w;
                if( w < FLT_EPSILON )
                    break;
                Vec3f mu = mptr[k].mean;
                Vec3f var = mptr[k].var;
                Vec3f diff = pix - mu;
                float d2 = diff.dot(diff);
                if( d2 < vT*(var[0] + var[1] + var[2]) )
                {
                    wsum -= w;
                    float dw = alpha*(1.f - w);
                    mptr[k].weight = w + dw;
                    mptr[k].mean = mu + alpha*diff;
                    var = Vec3f(std::max(var[0] + alpha*(diff[0]*diff[0] - var[0]), minVar),
                            std::max(var[1] + alpha*(diff[1]*diff[1] - var[1]), minVar),
                            std::max(var[2] + alpha*(diff[2]*diff[2] - var[2]), minVar));
                    mptr[k].var = var;
                    mptr[k].sortKey = w/std::sqrt(var[0] + var[1] + var[2]);

                    for( k1 = k-1; k1 >= 0; k1-- )
                    {
                        if( mptr[k1].sortKey >= mptr[k1+1].sortKey )
                            break;
                        std::swap( mptr[k1], mptr[k1+1] );
                    }

                    kHit = k;
                    break;
                }
            }

            if( kHit < 0 ) // no appropriate gaussian mixture found at all, remove the weakest mixture and create a new one
            {
                kHit = k = std::min(k, K-1);
                wsum += w0 - mptr[k].weight;
                mptr[k].weight = w0;
                mptr[k].mean = pix;
                mptr[k].var = Vec3f(var0, var0, var0);
                mptr[k].sortKey = sk0;


            }
            else
                for( ; k < K; k++ )
                    wsum += mptr[k].weight;

            float wscale = 1.f/wsum;
            wsum = 0;
            for( k = 0; k < K; k++ )
            {
                wsum += mptr[k].weight *= wscale;
                mptr[k].sortKey *= wscale;
                if( wsum > T && kForeground < 0 )
                    kForeground = k+1;
            }

            dst[0] = (uchar)(-(kHit >= kForeground));
            bgmean[0] = mptr[0].mean[0];
            bgmean[1] = mptr[0].mean[1];
            bgmean[2] = mptr[0].mean[2];
        }

    }

    medianBlur(fgmask,fgmask,5);
}

void BGModelMOG::process(Mat &image, Mat &fgmask, Mat &background, float learningRate)
{
    if(prev_img.empty())
        initialize(image.size(), image.type());

    fgmask.create(image.size(), CV_8U);
    background.create(frameSize,frameType);
    background = Scalar::all(0);
    fgmask = Scalar::all(0);

    ++nframes;
    learningRate = learningRate >= 0 && nframes > 1 ? learningRate : 1./std::min( 2*nframes, history);
    updateModel(image,fgmask,background,learningRate);

    prev_img = image;
}

}
