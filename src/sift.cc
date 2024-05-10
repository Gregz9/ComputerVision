#include "sift.h"
#include "filters.h"

void computeGaussianPyramid(cv::Mat *img){

  img = cv::resize(*img.rows*2, *img.cols*2, cv::INTER_LINEAR);
  // Computing the standard deviation of the gaussian kernel
  // For the very first image, init_sigma is equivalent to: 
  float std_dev = std::sqrt((MIN_SIGMA*MIN_SIGMA) - (INP_SIGMA*INP_SIGMA)); 
  std_dev /= PX_DST_MIN;

  cv::Mat kernel = create1DGaussKernel(std_dev);
  cv::sepFilter2D(*img, *img, -1, kernel, kernel);
  
  int num_scales = NUM_SCL_PER_OCT + 3; 
  float min_coeff = MIN_SIGMA / PX_DST_MIN;
  int k = std::pow(k, 1/num_scales);

  std::vector<float> std_devs {std_dev};
  GaussianPyramid pyramid; 
  pyramid.num_scales_per_oct = num_scales;

  // Computing standard deviations
  for (int i=1; i<num_scales; ++i){
    prev_sigma = std::pow(k, i-1) * min_coeff;
    sigma = k * prev_sigma; 
    std_devs.push_back(std::sqrt((sigma*sigma) - (prev_sigma*prev_sigma)));
  }
  
  pyramid.imgs.reserve(num_scales*NUM_OCT);
  for (int i=0; i<NUM_OCT; ++i){
    pyramid.imgs.push_back(*img);
    for(int j=0; j<num_scales; ++j){
      cv::Mat scale_img = pyramid.imgs.at((i*num_scales)-1);
      
    }
  } 

}
