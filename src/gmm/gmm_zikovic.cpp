#include "gmm_zikovic.h"
#include <bgs_util/surface_pyramid.h>

namespace GMM_ZIKOVIC
{

BGModelMOG::BGModelMOG()
{
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

void BGModelMOG::initialize(Size _frameSize, int _frameType)
{
    frameSize = _frameSize;
    frameType = _frameType;
    nframes = 0;

    int nchannels = CV_MAT_CN(frameType);
    CV_Assert( nchannels <= CV_CN_MAX );
    CV_Assert( nmixtures <= 255);

    bgmodel.create(1, frameSize.height*frameSize.width*nmixtures*(2 + nchannels), CV_32F );
    bgmodelUsedModes.create(frameSize,CV_8U);
    bgmodelUsedModes = Scalar::all(0);
    update_mask.create(frameSize,CV_8U);
    update_mask = Scalar::all(0);

    gmm0 = bgmodel.ptr<GMM>();
    mean0 = (float*)(bgmodel.ptr() + sizeof(GMM)*nmixtures*frameSize.height*frameSize.width);
    modesUsed0 = bgmodelUsedModes.ptr();
}

void BGModelMOG::initialize(Size _frameSize, int _frameType, Intrinsic &intrinsic)
{
    initialize(_frameSize, _frameType);
    transformer.reset(new ImageTransformer(intrinsic));
    transformer->setImageDimension(frameSize.width, frameSize.height);
}

void BGModelMOG::updateModel(Mat &image, Mat &fgmask, Mat &bgimg, float learningRate)
{
    float alphaT = learningRate;
    float alpha1 = 1.0 - alphaT;

    GMM* gmm = gmm0;
    float* mean = mean0;
    uchar* modesUsed = modesUsed0;
    uchar* mask = fgmask.ptr();
    uchar* bgmean = bgimg.ptr();

    int ncols = image.cols, nchannels = image.channels();
    float dData[CV_CN_MAX];

    Mat img;
    image.convertTo(img,CV_32FC(nchannels));
    float *data = img.ptr<float>();

    for(int y=0; y<image.rows; y++)
    {
        for(int x=0; x<image.cols; x++,data += nchannels, gmm += nmixtures, mean += nmixtures*nchannels,
                                   mask++, modesUsed++, bgmean+=nchannels)
        {
            bool background = false;
            bool fitsPDF = false;
            int nmodes = modesUsed[0];
            float totalWeight = 0.f;

            float* mean_m = mean;

            //go through all modes
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

                    //background? - Tb - usually larger than Tg
                    if( totalWeight < backgroundRatio && dist2 < varThreshold*var )
                        background = true;

                    //check fit
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
                        //limit the variance
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
            modesUsed[0] = uchar(nmodes);
            mask[0] = background ? 0 : 255;
            bgmean[0] = mean[0];
            bgmean[1] = mean[1];
            bgmean[2] = mean[2];
        }
    }

    medianBlur(fgmask,fgmask,5);
}

void BGModelMOG::warpModel(Mat &image, Mat &depth, Mat &fgmask, Eigen::Affine3d &transform,float learningRate)
{
    int nchannels = image.channels();
    int ncols = image.cols;
    float alphaT = learningRate;
    float alpha1 = 1.0 - alphaT;

    Mat img;
    image.convertTo(img,CV_32FC(nchannels));
    float *data = img.ptr<float>();

    Mat newmodel(1, frameSize.height*frameSize.width*nmixtures*(2 + nchannels), CV_32F,Scalar::all(0));
    Mat newmodes(frameSize,CV_8U,Scalar::all(0));

    GMM* gmm = newmodel.ptr<GMM>();
    float* mean = (float*)(newmodel.ptr() + sizeof(GMM)*nmixtures*image.rows*image.cols);
    uchar* modesUsed = newmodes.ptr();
    uchar* mask = fgmask.ptr();
    GMM* gmmOld0 = gmm0;
    float* meanOld0 = mean0;
    uchar* modesUsedOld0 = modesUsed0;

    float dData[CV_CN_MAX];

    for(int y=0; y<image.rows; y++)
    {
        for(int x=0; x<image.cols; x++, data += nchannels, gmm += nmixtures, mean += nmixtures*nchannels,
                                   modesUsed++, mask++)
        {
            double z = (double)depth.at<float>(y,x);
            int warped_x, warped_y;
            bool canWarp = transformer->transformPoint(x,y,z,warped_x,warped_y, transform);
            bool background = false;

            if(!canWarp)
            {
//                //initialize model with current value
                modesUsed[0] = uchar(1);
                gmm[0].weight = 1.0f;
                gmm[0].var = fVarInit;
                for( int c = 0; c < nchannels; c++)
                    mean[c] = data[c];
                background = true;
            }
            else
            {
//                //copying of gmm model to new locations
                GMM* gmmOld = gmmOld0 + nmixtures*(warped_y * ncols + warped_x);
                float* meanOld = meanOld0 + nmixtures*nchannels*(warped_y * ncols + warped_x);
                uchar* modesUsedOld = modesUsedOld0 + ncols*warped_y + warped_x;

                modesUsed[0] = modesUsedOld[0];
                int nmodes = modesUsed[0];
                float* mean_m = mean;
                float totalWeight = 0;
                bool fitsPDF = false;

                for(int mode=0; mode<nmodes; mode++, mean_m+=nchannels, meanOld+=nchannels)
                {
                    int swap_count = 0;
                    gmm[mode].weight = gmmOld[mode].weight;
                    gmm[mode].var = gmmOld[mode].var;

                    float var = gmm[mode].var;
                    float weight = alpha1 * gmm[mode].weight;
                    for(int c=0; c<nchannels; c++)
                        mean_m[c] = meanOld[c];

                    if(!fitsPDF)
                    {
                        dData[0] = mean_m[0] - data[0];
                        dData[1] = mean_m[1] - data[1];
                        dData[2] = mean_m[2] - data[2];
                        float dist2 = dData[0]*dData[0] + dData[1]*dData[1] + dData[2]*dData[2];

                        if(totalWeight < backgroundRatio && dist2 < varThreshold*var)
                            background = true;

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
                            //limit the variance
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
                    }

                    gmm[mode-swap_count].weight = weight;
                    totalWeight += weight;

                } //loop through all modes

                totalWeight = 1.f/totalWeight;
                for( int mode = 0; mode < nmodes; mode++ )
                {
                    gmm[mode].weight *= totalWeight;
                }

            }
            mask[0] = background ? 0 : 255;
        }
    }

    fgmask.copyTo(update_mask);
    medianBlur(fgmask,fgmask,5); //median filter to remove lines in segmentation map due to object boundary pixel misclassified

    //substitute old model with the new model
    bgmodel = newmodel.clone();
    bgmodelUsedModes = newmodes.clone();
    gmm0 = bgmodel.ptr<GMM>();
    mean0 = (float*)(bgmodel.ptr() + sizeof(GMM)*nmixtures*frameSize.height*frameSize.width);
    modesUsed0 = bgmodelUsedModes.ptr();
}

void BGModelMOG::updateWarpedModel(Mat &image, Mat &fgmask, Mat &bgimg, float learningRate)
{

//    Mat gradientMap;
//    Laplacian(image,gradientMap,frameType);
//    imshow("gradient",gradientMap);

    float alphaT = learningRate;
    float alpha1 = 1.0 - alphaT;
    int ncols = image.cols, nchannels = image.channels();

    Mat img;
    image.convertTo(img,CV_32FC(nchannels));
    float *data = img.ptr<float>();

    GMM* gmm = gmm0;
    float* mean = mean0;
    uchar* modesUsed = modesUsed0;
    uchar* mask_fg = fgmask.ptr();
    uchar* mask_update = update_mask.ptr();
    uchar* bgmean = bgimg.ptr();

    for(int y=0; y<image.rows; y++)
    {
        for(int x=0; x<image.cols; x++,data += nchannels, gmm += nmixtures, mean += nmixtures*nchannels,
                                   modesUsed++, mask_fg++, mask_update++, bgmean += nchannels)
        {
            //compare segmentation map before and after median filtering.
            //If do not agree, pixel is misclassified as moving pixel, need to update
            bool can_update = mask_update[0] != 0 && mask_fg[0] == 0;
            int nmodes = modesUsed[0];

            if(can_update)
            {
//                // replace the weakest or add a new one
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

//                // init
                for( int c = 0; c < nchannels; c++ )
                    mean[mode*nchannels + c] = data[c];

                gmm[mode].var = fVarInit;

//                //sort
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

            modesUsed[0] = uchar(nmodes);
            bgmean[0] = mean[0];
            bgmean[1] = mean[1];
            bgmean[2] = mean[2];
        }

    }

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
//    showBackgroundImage();
    showDifferenceImage(background,image);
    prev_img = image;
}

void BGModelMOG::process(Mat &image, Mat &depth_raw, Eigen::Affine3d &transform, Mat &fgmask, Mat &background, float learningRate)
{
    if(prev_img.empty())
        initialize(image.size(), image.type());

    cv::Mat depth;
    if(depth_raw.type() == CV_16UC1)
        SurfacePyramid::convertRawDepthImageSse(depth_raw, depth, 0.0002);
    else
        depth = depth_raw;

    fgmask.create(image.size(), CV_8U);
    background.create(frameSize,frameType);
    background = Scalar::all(0);
    fgmask = Scalar::all(0);

    ++nframes;
    learningRate = learningRate >= 0 && nframes > 1 ? learningRate : 1./std::min( 2*nframes, history);
    warpModel(image,depth,fgmask,transform,learningRate);
    updateWarpedModel(image,fgmask, background,learningRate);
//    showBackgroundImage();
    showDifferenceImage(background,image);
    prev_img = image;
}

void BGModelMOG::showBackgroundImage()
{
    int nchannels = 3;
    Mat bg1(frameSize,frameType,Scalar::all(0));
    Mat bg2(frameSize,frameType,Scalar::all(0));
    Mat bg3(frameSize,frameType,Scalar::all(0));
    Mat bg4(frameSize,frameType,Scalar::all(0));
    Mat bg5(frameSize,frameType,Scalar::all(0));


    float *mean = (float*)(bgmodel.ptr() + sizeof(GMM)*nmixtures*frameSize.width*frameSize.height);
    uchar* modesUsed = bgmodelUsedModes.ptr();

    for(int y=0; y<frameSize.height; y++)
    {
        for(int x=0; x<frameSize.width; x++, mean+=nmixtures*nchannels, modesUsed++)
        {

            bg1.at<Vec3b>(y,x) = Vec3b(mean[0],mean[1],mean[2]);
            bg2.at<Vec3b>(y,x) = Vec3b(mean[3],mean[4],mean[5]);
            bg3.at<Vec3b>(y,x) = Vec3b(mean[6],mean[7],mean[8]);
            bg4.at<Vec3b>(y,x) = Vec3b(mean[9],mean[10],mean[11]);
            bg5.at<Vec3b>(y,x) = Vec3b(mean[12],mean[13],mean[14]);

        }
    }

//    imshow("bg1",bg1);
    imshow("bg2",bg2);
    imshow("bg3",bg3);
    imshow("bg4",bg4);
    imshow("bg5",bg5);

}

void BGModelMOG::showDifferenceImage(Mat &img1, Mat &img2)
{
    Mat diffImage;

    absdiff(img1,img2,diffImage);
//    imshow("difference", diffImage);
}




}
