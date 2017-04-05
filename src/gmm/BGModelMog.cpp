#include "BGModelMog.h"

namespace BGS_LB
{

BGModelMog::BGModelMog()
{
    frameSize = cv::Size(0,0);
    frameType = 0;

    nmixtures = defaultNMixtures;
    varThreshold = defaultVarThreshold;
    backgroundRatio = defaultBackgroundRatio;
    initialVar = defaultInitialVar;
    learningRate = defaultLearningRate;
}


void BGModelMog::initialize(cv::Mat img_in)
{
    frameSize = img_in.size();
    frameType = img_in.type();

    CV_Assert( CV_MAT_DEPTH(frameType) == CV_8U );

    bgmodelMean = img_in.clone();

    int img_size = img_in.cols * img_in.rows;
    bgmodel.resize(img_size * nmixtures);
    bgmodeUsed = std::vector<int>(img_size, 1);
    std::vector<GMM>::iterator it = bgmodel.begin();

    for(int y=0; y<img_in.rows; y++)
    {
        const uchar* src = img_in.ptr<uchar>(y);

        for(int x=0; x<img_in.cols; x++, it += nmixtures)
        {
            cv::Vec3f pix(src[x*3], src[x*3+1], src[x*3+2]);

            it->mean = pix;
            it->var = cv::Vec3f(initialVar,initialVar,initialVar);
            it->weight = 1;
            it->sortKey = 1 / std::sqrt(cv::sum(it->var)[0]);
        }
    }

    //    bgmodel = new GMM[nmixtures*img_size];
    //    bgmodeUsed = new int[img_size];

    //    GMM *pMOG = bgmodel;
    //    int *pK = bgmodeUsed;

    //    int n=0;
    //    for(int y=0; y<img_in.rows; y++)
    //    {
    //        const uchar* src = img_in.ptr<uchar>(y);
    //        for(int x=0; y<img_in.cols; x++, n++, pMOG+=nmixtures)
    //        {
    //            pK[n] = 1;

    //            cv::Vec3f pix(src[x*3], src[x*3+1], src[x*3+2]);
    //            pMOG[0].mean = pix;
    //            pMOG[0].var = cv::Vec3f(initialVar,initialVar,initialVar);
    //            pMOG[0].weight = 1;
    //            pMOG[0].sortKey = pMOG[0].weight/sqrt(cv::sum(pMOG[0].var)[0]);
    //        }
    //    }

}

//void BGModelMog::updateModel(cv::Mat &img_in, cv::Mat &fgmask)
//{
//    int kBG;

//    GMM *pMOG = bgmodel;
//    int *pK = bgmodeUsed;

//    int n = 0;
//    for(int y = 0; y < img_in.rows; y++)
//    {
//        const uchar* src = img_in.ptr<uchar>(y);
//        uchar* dst = fgmask.ptr<uchar>(y);
//        uchar* bgmean = bgmodelMean.ptr<uchar>(y);

//        for(int x = 0; x < img_in.cols; x++, n++, pMOG+=nmixtures)
//        {
//            cv::Vec3f pix(src[x*3], src[x*3+1], src[x*3+2]);
//            int kHit = -1;

//            for(int k = 0; k < pK[n]; k++)
//            {
//                double dr = pix[0] - pMOG[k].mean[0];
//                double dg = pix[0] - pMOG[k].mean[1];
//                double db = pix[0] - pMOG[k].mean[2];
//                double d2 = dr*dr/pMOG[k].var[0] + dg*dg/pMOG[k].var[1] + db*db/pMOG[k].var[2];

//                if(d2 < varThreshold)
//                {
//                    kHit = k;
//                    break;
//                }
//            }

//            // Adjust parameters

//            // matching distribution found
//            if(kHit != -1)
//            {
//                for(int k = 0; k < pK[n]; k++)
//                {
//                    if(k == kHit)
//                    {
//                        pMOG[k].weight = pMOG[k].weight + learningRate*(1.0f - pMOG[k].weight);

//                        double d;

//                        d = pix[0] - pMOG[k].mean[0];
//                        if(d*d > DBL_MIN)
//                            pMOG[k].mean[0] += learningRate*d;

//                        d = pix[1] - pMOG[k].mean[1];
//                        if(d*d > DBL_MIN)
//                            pMOG[k].mean[1] += learningRate*d;

//                        d = pix[2] - pMOG[k].mean[2];
//                        if(d*d > DBL_MIN)
//                            pMOG[k].mean[2] += learningRate*d;

//                        d = (pix[0] - pMOG[k].mean[0])*(pix[0] - pMOG[k].mean[0]) - pMOG[k].var[0];
//                        if(d*d > DBL_MIN)
//                            pMOG[k].var[0] += learningRate*d;

//                        d = (pix[1] - pMOG[k].mean[1])*(pix[1] - pMOG[k].mean[1]) - pMOG[k].var[1];
//                        if(d*d > DBL_MIN)
//                            pMOG[k].var[1] += learningRate*d;

//                        d = (pix[2] - pMOG[k].mean[2])*(pix[2] - pMOG[k].mean[2]) - pMOG[k].var[2];
//                        if(d*d > DBL_MIN)
//                            pMOG[k].var[2] += learningRate*d;

//                        pMOG[k].var[0] = (std::max)(pMOG[k].var[0],initialVar);
//                        pMOG[k].var[1] = (std::max)(pMOG[k].var[1],initialVar);
//                        pMOG[k].var[2] = (std::max)(pMOG[k].var[2],initialVar);
//                    }
//                    else
//                        pMOG[k].weight = (1.0 - learningRate)*pMOG[k].weight;
//                }
//            }
//            // no match found... create new one
//            else
//            {
//                if(pK[n] < nmixtures)
//                    pK[n]++;

//                kHit = pK[n] - 1;

//                if(pK[n] == 1)
//                    pMOG[kHit].weight = 1.0;
//                else
//                    pMOG[kHit].weight = learningRate;

//                pMOG[kHit].mean = pix;
//                pMOG[kHit].var = cv::Vec3f(initialVar,initialVar,initialVar);
//            }

//            double wsum = 0.0;

//            for(int k = 0; k < pK[n]; k++)
//                wsum += pMOG[k].weight;

//            double wfactor = 1.0/wsum;

//            for(int k = 0; k < pK[n]; k++)
//            {
//                pMOG[k].weight *= wfactor;
//                pMOG[k].sortKey = pMOG[k].weight/sqrt(pMOG[k].var[0]+pMOG[k].var[1]+pMOG[k].var[2]);
//            }

//            // Sort distributions

//            for (int k = 0; k < kHit; k++)
//            {
//                if(pMOG[kHit].sortKey > pMOG[k].sortKey)
//                {
//                    std::swap(pMOG[kHit],pMOG[k]);
//                    break;
//                }
//            }

//            // Determine background distributions

//            wsum = 0.0;

//            for(int k = 0; k < pK[n]; k++)
//            {
//                wsum += pMOG[k].weight;

//                if(wsum > backgroundRatio)
//                {
//                    kBG = k;
//                    break;
//                }
//            }

//            dst[x] = kHit > kBG ? 255 : 0;
//            bgmean[x*3] = (uchar)pMOG[0].mean[0];
//            bgmean[x*3+1] = (uchar)pMOG[0].mean[1];
//            bgmean[x*3+1] = (uchar)pMOG[0].mean[2];
//        }
//    }
//}

void BGModelMog::updateModel(cv::Mat &img_in, cv::Mat &fgmask)
{
    int n = 0;
    std::vector<GMM>::iterator gmm_it = bgmodel.begin();

    for(int y=0; y<img_in.rows; y++)
    {
        const uchar* src = img_in.ptr<uchar>(y);
        uchar* dst = fgmask.ptr<uchar>(y);

        for(int x=0; x<img_in.cols; x++, n++, gmm_it += nmixtures)
        {
            cv::Vec3f pix(src[x*3], src[x*3+1], src[x*3+2]);
            int kHit = -1;

            for(int k=0; k<bgmodeUsed[n]; k++)
            {
                GMM &model = gmm_it[k];
//                cv::Vec3f diff = pix - model.mean;
//                cv::Vec3f temp;
//                cv::multiply(diff,diff,temp);
//                cv::divide(temp,model.var,temp);
//                float d2 = cv::sum(temp)[0];
                float dr = pix[0] - model.mean[0];
                float dg = pix[1] - model.mean[1];
                float db = pix[2] - model.mean[2];
                float d2 = dr*dr/model.var[0] + dg*dg/model.var[1] + db*db/model.var[2];

                if(d2 < varThreshold)
                {
                    kHit = k;
                    break;
                }
            }

            if(kHit != -1)
            {
                for(int k=0; k<bgmodeUsed[n]; k++)
                {
                    GMM &model = gmm_it[k];
                    if(k == kHit)
                    {
                        model.weight = model.weight + learningRate * (1 - model.weight);
//                        cv::Vec3f diff = pix - model.mean;
//                        model.mean += learningRate * diff;

//                        cv::Vec3f diffVar;
//                        cv::multiply(diff,diff,diffVar);
//                        diffVar = diffVar - model.var;
//                        model.var += learningRate * diffVar;
                        float d;

                        d = pix[0] - model.mean[0];
                        if(d*d > DBL_MIN)
                            model.mean[0] += learningRate * d;

                        d = d*d - model.var[0];
                        if(d*d > DBL_MIN)
                            model.var[0] += learningRate * d;

                        d = pix[1] - model.mean[1];
                        if(d*d > DBL_MIN)
                            model.mean[1] += learningRate * d;

                        d = d*d - model.var[1];
                        if(d*d > DBL_MIN)
                            model.var[1] += learningRate * d;

                        d = pix[2] - model.mean[2];
                        if(d*d > DBL_MIN)
                            model.mean[2] += learningRate * d;

                        d = d*d - model.var[2];
                        if(d*d > DBL_MIN)
                            model.var[2] += learningRate * d;

                        model.var[0] = std::max(model.var[0], initialVar);
                        model.var[1] = std::max(model.var[1], initialVar);
                        model.var[2] = std::max(model.var[2], initialVar);
                    }

                    else
                        model.weight = (1 - learningRate) * model.weight;
                }
            }

            else
            {
                if(bgmodeUsed[n] < nmixtures)
                    bgmodeUsed[n]++;

                kHit = bgmodeUsed[n] - 1;
                GMM &model = gmm_it[kHit];
                if(bgmodeUsed[n] == 1)
                    model.weight = 1;
                else
                    model.weight = learningRate;

                model.mean = pix;
                model.var = cv::Vec3f(initialVar,initialVar,initialVar);
            }

            //normalize weight
            float wsum = 0;
            for(int k=0; k<bgmodeUsed[n]; k++)
                wsum += gmm_it[k].weight;

            float wscale = 1.0 / wsum;
            for(int k=0; k<bgmodeUsed[n]; k++)
            {
                GMM &model = gmm_it[k];
                model.weight *= wscale;
                model.sortKey = model.weight / sqrt(cv::sum(model.var)[0]);
            }

            //sort
            for(int k=0; k<kHit; k++)
            {
                if(gmm_it[kHit].sortKey > gmm_it[k].sortKey)
                {
                    std::swap(gmm_it[kHit], gmm_it[k]);
                    break;
                }
            }

            wsum = 0;
            int kBG;
            for(int k=0; k<bgmodeUsed[n]; k++)
            {
                wsum += gmm_it[k].weight;
                if(wsum > backgroundRatio)
                {
                    kBG = k;
                    break;
                }
            }

            dst[x] = kHit > kBG ? 255 : 0;
        }
    }
}

void BGModelMog::process(cv::Mat &img_in, cv::Mat &foreground, cv::Mat &background)
{
    if(prev_img.empty())
        initialize(img_in);

    foreground.create(img_in.size(), CV_8U );
    updateModel(img_in,foreground);
    background = bgmodelMean;
    prev_img = img_in;
}


} //BGS



