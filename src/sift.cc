#include "sift.h"
#include "filters.h"

GaussianPyramid computeGaussianPyramid(cv::Mat img){
  cv::resize(img, img, cv::Size(img.rows*2, img.cols*2), cv::INTER_LINEAR);
  // Computing the standard deviation of the gaussian kernel
  // For the very first image, init_sigma is equivalent to: 
  float std_dev = std::sqrt((MIN_SIGMA*MIN_SIGMA) - (INP_SIGMA*INP_SIGMA)); 
  std_dev /= PX_DST_MIN;

  cv::Mat kernel = create1DGaussKernel(std_dev);
  cv::sepFilter2D(img, img, -1, kernel, kernel);
  
  int num_scales = NUM_SCL_PER_OCT + 3; 
  float min_coef = MIN_SIGMA / PX_DST_MIN;
  auto k = static_cast<float>(std::pow(2, 1.0/num_scales));

  std::vector<float> std_devs {std_dev};
  GaussianPyramid pyramid;
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

    for(int j=1; j < (int)std_devs.size();j++){
      const cv::Mat& scale_img = pyramid.imgs.at(i*num_scales);
      cv::sepFilter2D(scale_img, scale_img, -1, std_devs[j], std_devs[j]);
      pyramid.imgs.push_back(scale_img.clone());
    }
    const cv::Mat& next_img = pyramid.imgs.at((i*num_scales)+NUM_SCL_PER_OCT);
    cv::resize(next_img, img, cv::Size(img.rows/2, img.cols/2), cv::INTER_LINEAR);

  }
return pyramid;
}
