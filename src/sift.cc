#include "sift.h"
#include "filters.h"
#include <cmath>
Pyramid computeGaussianPyramid(cv::Mat img){
    cv::resize(img, img, cv::Size(img.rows*2, img.cols*2), cv::INTER_LINEAR);
    // Computing the standard deviation of the gaussian kernel
    // For the very first image, init_sigma is equivalent to:
    float std_dev = std::sqrt((MIN_SIGMA*MIN_SIGMA) - (INP_SIGMA*INP_SIGMA));
    std_dev /= PX_DST_MIN;

    cv::Mat kernel = create1DGaussKernel(std_dev);
    cv::sepFilter2D(img, img, -1, kernel, kernel);

    int num_scales = NUM_SCL_PER_OCT + 3;
    float min_coef = MIN_SIGMA / PX_DST_MIN;
    auto k = static_cast<float>(std::pow(2, 1.0/NUM_SCL_PER_OCT));

    std::vector<float> std_devs {min_coef};
    Pyramid pyramid;
    pyramid.num_scales_per_oct = num_scales;

    // Computing standard deviations
    for (int i=1; i<num_scales; ++i){
        auto prev_sigma = static_cast<float>(std::pow(k, i-1) * min_coef);
        float sigma = k * prev_sigma;
        std_devs.push_back(std::sqrt((sigma*sigma) - (prev_sigma*prev_sigma)));
    }

    pyramid.imgs.reserve(num_scales*NUM_OCT);
    for (int i=0; i<NUM_OCT; i++){
        pyramid.imgs.push_back(img.clone());

        for(int j=1; j < (int)std_devs.size(); j++){
            const cv::Mat& scale_img = pyramid.imgs.at(i*num_scales);
            cv::Mat blur_kernel = create1DGaussKernel(std_dev);
            cv::sepFilter2D(scale_img, scale_img, CV_32F, blur_kernel, blur_kernel);
            pyramid.imgs.push_back(scale_img.clone());
        }
        const cv::Mat& next_img = pyramid.imgs.at((i*num_scales)+NUM_SCL_PER_OCT);
        cv::resize(next_img, img, cv::Size(img.rows/2, img.cols/2), cv::INTER_LINEAR);

    }
    return pyramid;
}

Pyramid computeDoGPyramid(const Pyramid gauss) {

    Pyramid dog;
    dog.num_scales_per_oct = gauss.num_scales_per_oct - 1;
    dog.imgs.reserve(dog.num_scales_per_oct*dog.num_oct);

    for (int o = 0; o < dog.num_oct; ++o) {
        for (int s = 1; s < gauss.num_scales_per_oct; ++s) { // Start from s = 0
            cv::Mat diff_img = gauss.imgs[(o * gauss.num_scales_per_oct) + s] - gauss.imgs[(o * gauss.num_scales_per_oct) + s-1];

            dog.imgs.push_back(diff_img);

        }
    }
    return dog;
}

keypoints locateExtrema(const Pyramid dog, float C_dog, float C_edge) {
    // Impossible to know how many keypoint candidates there'll
    // be, hence cannot reserve a specific size for the vector.
    keypoints k_points{};
    for(int o = 0; o < dog.num_oct; ++o) {
        // DoG pyramid has already num_scales-1, hence only 1 subtracted
        for(int s=1; s < dog.num_scales_per_oct-1; ++s) {
            cv::Mat img_DoG = dog.imgs.at((o*dog.num_scales_per_oct)+s);
            cv::Mat prev_img_DoG = dog.imgs.at((o*dog.num_scales_per_oct)+s-1);
            cv::Mat next_img_DoG = dog.imgs.at((o*dog.num_scales_per_oct)+s+1);
            for(int i = 1; i < img_DoG.rows -1; ++i) {
                for(int j =1; j < img_DoG.cols -1; ++j) {
                    float current_pixel = img_DoG.at<float>(i,j);
                    // It's more efficient to check for the threshold criteria here, rather than later.
                    if(std::abs(img_DoG.at<float>(i,j)) < 0.8*C_dog) {
                        continue;
                    }

                    bool max = true, min = true;

                    for(int x = -1; x<2; ++x) {
                        for (int y = -1; y < 2; ++y) {
                            float neighbor_pixel_curr = img_DoG.at<float>(i + x, j + y);
                            float neighbor_pixel_prev = prev_img_DoG.at<float>(i + x, j + y);
                            float neighbor_pixel_next = next_img_DoG.at<float>(i + x, j + y);

                            if (current_pixel < neighbor_pixel_prev || current_pixel < neighbor_pixel_curr ||
                                current_pixel < neighbor_pixel_next) {
                                max = false;
                            }
                            if (current_pixel > neighbor_pixel_prev || current_pixel > neighbor_pixel_curr ||
                                current_pixel > neighbor_pixel_next) {
                                min = false;
                            }
                        }
                    }
                    if (max || min) {
                        KeyPoint k = {i, j, o, s};
                        k_points.push_back(k);
                    }
                }
            }
        }
    }
    return k_points;
}