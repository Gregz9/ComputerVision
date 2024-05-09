#include "filters.h"
#include <atomic>

cv::Mat create1DGaussKernel(float std_dev)
{
  int radius = static_cast<int>(std::ceil(3.5f * std_dev)); 
  
  const int length = 2*radius + 1; 
  cv::Mat kernel(length, 1, CV_32F); 

  const float factor = -0.5f / (std_dev*std_dev);
  float sum = 0.f;

  for(int i = 0; i < length; ++i) {
    const float x = static_cast<float>(i - radius);
    const float kernel_element = std::exp(x*x * factor); 
    kernel.at<float>(i) = kernel_element; 
    sum += kernel_element;
  }

  return kernel/sum;
}

cv::Mat create1DDerivateGaussKernel(float std_dev) {
  
  int radius = static_cast<int>(std::ceil(3.5f * std_dev)); 
  cv::Mat kernel = create1DGaussKernel(std_dev);

  const int length = kernel.rows; 
  const float factor = -1.f / (std_dev*std_dev);
  
  for(int i = 0; i < length; ++i)
  {
    const float x = static_cast<float>(i - radius);
    kernel.at<float>(i) *= x * factor; 
  }

  return kernel;

}

