#pragma once

#include "opencv2/core.hpp"
#include <stdlib.h>
#include <iostream>

cv::Mat create1DGaussKernel(double std_dev);

cv::Mat create1DDerivateGaussKernel(double std_dev);

