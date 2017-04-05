#include "bgfg_gaussmix2.h"

namespace BGS_ZIKOVIC
{
BGModelMog::BGModelMog(Intrinsic intrinsic)
{
    transformer.reset(new ImageTransformer(intrinsic));
    hasTransformer = true;
    frameSize = Size(0,0);
    frameType = 0;

    nframes = 0;
    history = defaultHistory2;
    varThreshold = defaultVarThreshold2;

    nmixtures = defaultNMixtures2;
    backgroundRatio = defaultBackgroundRatio2;
    fVarInit = defaultVarInit2;
    fVarMax  = defaultVarMax2;
    fVarMin = defaultVarMin2;

    varThresholdGen = defaultVarThresholdGen2;
}

BGModelMog::BGModelMog()
{
    hasTransformer = false;
    frameSize = Size(0,0);
    frameType = 0;

    nframes = 0;
    history = defaultHistory2;
    varThreshold = defaultVarThreshold2;

    nmixtures = defaultNMixtures2;
    backgroundRatio = defaultBackgroundRatio2;
    fVarInit = defaultVarInit2;
    fVarMax  = defaultVarMax2;
    fVarMin = defaultVarMin2;

    varThresholdGen = defaultVarThresholdGen2;
}

void BGModelMog::initialize(Size _frameSize, int _frameType)
{
    frameSize = _frameSize;
    frameType = _frameType;
    nframes = 0;

    if(hasTransformer)transformer->setImageDimension(frameSize.width, frameSize.height);

    int nchannels = CV_MAT_CN(frameType);
    CV_Assert( nchannels <= CV_CN_MAX );
    CV_Assert( nmixtures <= 255);

    bgmodel.create(1, frameSize.height*frameSize.width*nmixtures*(2 + nchannels), CV_32F );
    bgmodelUsedModes.create(frameSize,CV_8U);
    bgmodelUsedModes = Scalar::all(0);
    updateMask = bgmodelUsedModes.clone();
}

void BGModelMog::updateModel(Mat &image, Mat &fgmask, Mat &bgimg, float learningRate)
{
    bgimg.create(frameSize,frameType);
    bgimg = Scalar::all(0);

    GMM* gmm0 = bgmodel.ptr<GMM>();
    float* mean0 = (float*)(bgmodel.ptr() + sizeof(GMM)*nmixtures*image.rows*image.cols);
    uchar* modesUsed0 = bgmodelUsedModes.ptr();
    float alphaT = learningRate;
    float alpha1 = 1.0 - alphaT;

    int ncols = image.cols, nchannels = image.channels();
    AutoBuffer<float> buf(image.cols*nchannels);
    float dData[CV_CN_MAX];

    for(int y=0; y<image.rows; y++)
    {
        const float* data = buf;

        image.row(y).convertTo(Mat(1, ncols, CV_32FC(nchannels), (void*)data), CV_32F);
        float* mean = mean0 + ncols*nmixtures*nchannels*y;
        GMM* gmm = gmm0 + ncols*nmixtures*y;
        uchar* modesUsed = modesUsed0 + ncols*y;
        uchar* mask = fgmask.ptr(y);

        for(int x=0; x<image.cols; x++,data += nchannels, gmm += nmixtures, mean += nmixtures*nchannels)
        {
            bool background = false;//return value -> true - the pixel classified as background

            //internal:
            bool fitsPDF = false;//if it remains zero a new GMM mode will be added
            int nmodes = modesUsed[x];//current number of modes in GMM
            float totalWeight = 0.f;

            float* mean_m = mean;

            //////
            //go through all modes
            for( int mode = 0; mode < nmodes; mode++, mean_m += nchannels )
            {
//                float weight = alpha1*gmm[mode].weight + prune;
                float weight = alpha1*gmm[mode].weight;
                int swap_count = 0;
                ////
                //fit not found yet
                if( !fitsPDF )
                {
                    //check if it belongs to some of the remaining modes
                    float var = gmm[mode].var;

                    //calculate difference and distance
                    float dist2;

                    dData[0] = mean_m[0] - data[0];
                    dData[1] = mean_m[1] - data[1];
                    dData[2] = mean_m[2] - data[2];
                    dist2 = dData[0]*dData[0] + dData[1]*dData[1] + dData[2]*dData[2];

                    //background? - Tb - usually larger than Tg
                    if( totalWeight < backgroundRatio && dist2 < varThreshold*var )
                        background = true;

                    //check fit
                    if( dist2 < varThresholdGen*var )
                    {
                        /////
                        //belongs to the mode
                        fitsPDF = true;

                        //update distribution

                        //update weight
                        weight += alphaT;
                        float k = alphaT/weight;

                        //update mean
                        for( int c = 0; c < nchannels; c++ )
                            mean_m[c] -= k*dData[c];

                        //update variance
                        float varnew = var + k*(dist2-var);
                        //limit the variance
                        varnew = MAX(varnew, fVarMin);
                        varnew = MIN(varnew, fVarMax);
                        gmm[mode].var = varnew;

                        //sort
                        //all other weights are at the same place and
                        //only the matched (iModes) is higher -> just find the new place for it
                        for( int i = mode; i > 0; i-- )
                        {
                            //check one up
                            if( weight < gmm[i-1].weight )
                                break;

                            swap_count++;
                            //swap one up
                            std::swap(gmm[i], gmm[i-1]);
                            for( int c = 0; c < nchannels; c++ )
                                std::swap(mean[i*nchannels + c], mean[(i-1)*nchannels + c]);
                        }
                        //belongs to the mode - bFitsPDF becomes 1
                    }
                }//!bFitsPDF)

                gmm[mode-swap_count].weight = weight;//update weight by the calculated value
                totalWeight += weight;
            }
            //go through all modes

            //renormalize weights
            totalWeight = 1.f/totalWeight;
            for( int mode = 0; mode < nmodes; mode++ )
            {
                gmm[mode].weight *= totalWeight;
            }

            //make new mode if needed and exit
            if( !fitsPDF && alphaT > 0.f )
            {
                // replace the weakest or add a new one
                int mode = nmodes == nmixtures ? nmixtures-1 : nmodes++;

                if (nmodes==1)
                    gmm[mode].weight = 1.f;
                else
                {
                    gmm[mode].weight = alphaT;

                    // renormalize all other weights
                    for( int i = 0; i < nmodes-1; i++ )
                        gmm[i].weight *= alpha1;
                }

                // init
                for( int c = 0; c < nchannels; c++ )
                    mean[mode*nchannels + c] = data[c];

                gmm[mode].var = fVarInit;

                //sort
                //find the new place for it
                for( int i = nmodes - 1; i > 0; i-- )
                {
                    // check one up
                    if( alphaT < gmm[i-1].weight )
                        break;

                    // swap one up
                    std::swap(gmm[i], gmm[i-1]);
                    for( int c = 0; c < nchannels; c++ )
                        std::swap(mean[i*nchannels + c], mean[(i-1)*nchannels + c]);
                }
            }

            //set the number of modes
            modesUsed[x] = uchar(nmodes);
            mask[x] = background ? 0 : 255;
            bgimg.at<Vec3b>(y,x) = Vec3b(mean[0],mean[1],mean[2]);
        }
    }

    medianBlur(fgmask,fgmask,5);
}

void BGModelMog::warpModel(Mat &image, Mat &depth, Mat &fgmask, Eigen::Affine3d &transform)
{
    int nchannels = image.channels();
    int ncols = image.cols;

    Mat newmodel(1, frameSize.height*frameSize.width*nmixtures*(2 + nchannels), CV_32F );
    Mat newmodes(frameSize,CV_8U,Scalar::all(0));

    GMM* gmm0 = newmodel.ptr<GMM>();
    float* mean0 = (float*)(newmodel.ptr() + sizeof(GMM)*nmixtures*image.rows*image.cols);
    uchar* modesUsed0 = newmodes.ptr();

    GMM* gmmOld0 = bgmodel.ptr<GMM>();
    float* meanOld0 = (float*)(bgmodel.ptr() + sizeof(GMM)*nmixtures*image.rows*image.cols);
    uchar* modesUsedOld0 = bgmodelUsedModes.ptr();

    AutoBuffer<float> buf(image.cols*nchannels);
    float dData[CV_CN_MAX];
    updateMask = Scalar::all(0);

    for(int y=0; y<image.rows; y++)
    {
        const float* data = buf;

        image.row(y).convertTo(Mat(1, ncols, CV_32FC(nchannels), (void*)data), CV_32F);
        float* mean = mean0 + ncols*nmixtures*nchannels*y;
        GMM* gmm = gmm0 + ncols*nmixtures*y;
        uchar* modesUsed = modesUsed0 + ncols*y;
        uchar* mask = updateMask.ptr(y);

        for(int x=0; x<image.cols; x++, data += nchannels, gmm += nmixtures, mean += nmixtures*nchannels)
        {
            double z = (double)depth.at<float>(y,x);
            int warped_x, warped_y;

            //calculate warped coordinate
            bool canWarp = transformer->transformPoint(x,y,z,warped_x,warped_y, transform);

            if(!canWarp)
            {
                //initialize model with current value
                modesUsed[x] = uchar(1);
                gmm[0].weight = 1.0f;
                gmm[0].var = fVarInit;
                for( int c = 0; c < nchannels; c++)
                    mean[c] = data[c];
            }
            else
            {
                //copying of gmm model to new locations
                GMM* gmmOld = gmmOld0 + nmixtures*(warped_y * ncols + warped_x);
                float* meanOld = meanOld0 + nmixtures*nchannels*(warped_y * ncols + warped_x);
                uchar* modesUsedOld = modesUsedOld0 + ncols*warped_y + warped_x;

                modesUsed[x] = modesUsedOld[0];
                float* mean_m = mean;
                float totalWeight = 0;
                bool checkForeground = true;

                for(int k=0; k<modesUsed[x]; k++, mean_m+=nchannels)
                {
                    gmm[k].weight = gmmOld[k].weight;

                    gmm[k].var = gmmOld[k].var;
                    for(int c=0; c<nchannels; c++)
                        mean_m[c] = meanOld[c];

                    float dist2;
                    if(checkForeground)
                    {
                        totalWeight += gmm[k].weight;
                        dData[0] = mean_m[0] - data[0];
                        dData[1] = mean_m[1] - data[1];
                        dData[2] = mean_m[2] - data[2];
                        dist2 = dData[0]*dData[0] + dData[1]*dData[1] + dData[2]*dData[2];
                    }


                    //label foreground
                    if( totalWeight > backgroundRatio || dist2 > varThreshold*gmm[k].var)
                    {
                        mask[x] = 255;
                        checkForeground = false;
                    }
                }
            }
        }
    }

    medianBlur(updateMask,updateMask,5);
    updateMask.copyTo(fgmask);

}

void BGModelMog::updateWarpedModel(Mat &image, Mat &bgimg, float learningRate)
{
    bgimg.create(frameSize,frameType);
    bgimg = Scalar::all(0);

    GMM* gmm0 = bgmodel.ptr<GMM>();
    float* mean0 = (float*)(bgmodel.ptr() + sizeof(GMM)*nmixtures*image.rows*image.cols);
    uchar* modesUsed0 = bgmodelUsedModes.ptr();
    float alphaT = learningRate;
    float alpha1 = 1.0 - alphaT;

    int ncols = image.cols, nchannels = image.channels();
    AutoBuffer<float> buf(image.cols*nchannels);
    float dData[CV_CN_MAX];

    for(int y=0; y<image.rows; y++)
    {
        const float* data = buf;

        image.row(y).convertTo(Mat(1, ncols, CV_32FC(nchannels), (void*)data), CV_32F);
        float* mean = mean0 + ncols*nmixtures*nchannels*y;
        GMM* gmm = gmm0 + ncols*nmixtures*y;
        uchar* modesUsed = modesUsed0 + ncols*y;
        uchar* mask = updateMask.ptr(y);

        for(int x=0; x<image.cols; x++,data += nchannels, gmm += nmixtures, mean += nmixtures*nchannels)
        {
            if(mask[x] != 0) continue;

            bool fitsPDF = false;//if it remains zero a new GMM mode will be added
            int nmodes = modesUsed[x];//current number of modes in GMM
            float totalWeight = 0;

            float* mean_m = mean;

            for( int mode = 0; mode < nmodes; mode++, mean_m += nchannels )
            {
                float weight = alpha1*gmm[mode].weight;
                int swap_count = 0;

                if( !fitsPDF )
                {
                    float var = gmm[mode].var;
                    float dist2;

                    dData[0] = mean_m[0] - data[0];
                    dData[1] = mean_m[1] - data[1];
                    dData[2] = mean_m[2] - data[2];
                    dist2 = dData[0]*dData[0] + dData[1]*dData[1] + dData[2]*dData[2];

                    if( dist2 < varThresholdGen*var )
                    {
                        fitsPDF = true;

                        //update weight
                        weight += alphaT;
                        float k = alphaT/weight;

                        //update mean
                        for( int c = 0; c < nchannels; c++ )
                            mean_m[c] -= k*dData[c];

                        //update variance
                        float varnew = var + k*(dist2-var);
                        varnew = MAX(varnew, fVarMin);
                        varnew = MIN(varnew, fVarMax);
                        gmm[mode].var = varnew;

                        //sort
                        for( int i = mode; i > 0; i-- )
                        {
                            //check one up
                            if( weight < gmm[i-1].weight )
                                break;

                            swap_count++;
                            //swap one up
                            std::swap(gmm[i], gmm[i-1]);
                            for( int c = 0; c < nchannels; c++ )
                                std::swap(mean[i*nchannels + c], mean[(i-1)*nchannels + c]);
                        }

                    }

                    gmm[mode-swap_count].weight = weight;//update weight by the calculated value
                    totalWeight += weight;
                }


            }

            //renormalize weights
            totalWeight = 1.f/totalWeight;
            for( int mode = 0; mode < nmodes; mode++ )
            {
                gmm[mode].weight *= totalWeight;
            }

            if(!fitsPDF)
            {
                // replace the weakest or add a new one
                int mode = nmodes == nmixtures ? nmixtures-1 : nmodes++;

                if (nmodes==1)
                    gmm[mode].weight = 1.f;
                else
                {
                    gmm[mode].weight = alphaT;

                    // renormalize all other weights
                    for( int i = 0; i < nmodes-1; i++ )
                        gmm[i].weight *= alpha1;
                }

                // init
                for( int c = 0; c < nchannels; c++ )
                    mean[mode*nchannels + c] = data[c];

                gmm[mode].var = fVarInit;

                //sort
                //find the new place for it
                for( int i = nmodes - 1; i > 0; i-- )
                {
                    // check one up
                    if( alphaT < gmm[i-1].weight )
                        break;

                    // swap one up
                    std::swap(gmm[i], gmm[i-1]);
                    for( int c = 0; c < nchannels; c++ )
                        std::swap(mean[i*nchannels + c], mean[(i-1)*nchannels + c]);
                }
            }

            modesUsed[x] = uchar(nmodes);
            bgimg.at<Vec3b>(y,x) = Vec3b(mean[0],mean[1],mean[2]);
        }

    }

}


void BGModelMog::process(Mat &image, Mat &fgmask, Mat &background, double learningRate)
{
    if(prev_img.empty())
        initialize(image.size(), image.type());

    fgmask.create(image.size(), CV_8U);
    ++nframes;
    learningRate = learningRate >= 0 && nframes > 1 ? learningRate : 1./std::min( 2*nframes, history);
    updateModel(image,fgmask,background,(float)learningRate);
    prev_img = image;
}

void BGModelMog::process(Mat &image, Mat &depth, Eigen::Affine3d &transform, Mat &fgmask, Mat &background, double learningRate)
{
    if(prev_img.empty())
        initialize(image.size(), image.type());

    ++nframes;
//    learningRate = learningRate >= 0 && nframes > 1 ? learningRate : 1./std::min( 2*nframes, history);
    warpModel(image,depth,fgmask,transform);
    updateWarpedModel(image,background,(float)learningRate);

    prev_img = image;
}


}
