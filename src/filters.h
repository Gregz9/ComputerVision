#pragma once

#include "opencv2/core.hpp"
#include <stdlib.h>
#include <iostream>

cv::Mat create1DGaussKernel(float std_dev);

cv::Mat create1DDerivateGaussKernel(float std_dev);

