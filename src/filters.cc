#include "filters.h"
#include <atomic>

cv::Mat create1DGaussKernel(double std_dev)
{
  int radius = static_cast<int>(std::ceil(3.5f * std_dev));

  const int length = 2*radius + 1;
  cv::Mat kernel(length, 1, CV_64F);

  const double factor = -0.5 / (std_dev*std_dev);
  double sum = 0.;

  for(int i = 0; i < length; ++i) {
    const double x = static_cast<double>(i - radius);
    const double kernel_element = std::exp(x*x * factor);
    kernel.at<double>(i) = kernel_element;
    sum += kernel_element;
  }

  return kernel/sum;
}

/*cv::Mat create1DGaussKernel(double std_dev)
{
    int size = std::ceil(6 * std_dev);
    if (size % 2 == 0)
        size++;
    int center = size / 2;
    cv::Mat kernel(size, 1, CV_64F);
    double sum = 0;
    for (int k = -center; k <= center; k++) {
        double val = std::exp(-(k * k) / (2 * std_dev * std_dev));
        kernel.at<double>(center + k) = val;
        sum += val;
    }
    for (int i = 0; i < size; ++i) {
        kernel.at<double>(i) /= sum;
    }
    return kernel;
}*/


cv::Mat create1DDerivateGaussKernel(double std_dev) {
  
  int radius = static_cast<int>(std::ceil(3.5f * std_dev)); 
  cv::Mat kernel = create1DGaussKernel(std_dev);

  const int length = kernel.rows; 
  const double factor = -1.f / (std_dev*std_dev);
  
  for(int i = 0; i < length; ++i)
  {
    const double x = static_cast<double>(i - radius);
    kernel.at<double>(i) *= x * factor; 
  }

  return kernel;

}

