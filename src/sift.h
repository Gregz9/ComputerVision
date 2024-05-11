#pragma once

// #include "opencv2/core.hpp"
#include <vector>
#include <opencv2/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/opencv.hpp>
#include <stdlib.h>

constexpr int NUM_OCT = 5; 
constexpr int NUM_SCL_PER_OCT = 3; 
constexpr float MIN_SIGMA = 0.8; // Minimal blur level
constexpr float INP_SIGMA = 0.5; // Assumed blur level of input image
constexpr float IN_PX_DST = 1.0; 
constexpr float PX_DST_MIN = 0.5;

constexpr int MAX_ITERS_REF = 5;
constexpr float MAX_REF_THR = 0.5; 
constexpr float DOG_THR = 0.015;
constexpr float EDGE_THR = 0.f;

struct GaussianPyramid{
  int num_oct = NUM_OCT;
  int num_scales_per_oct;
  std::vector<cv::Mat> imgs{}; 
};

void computeDogThr();

GaussianPyramid computeGaussianPyramid(cv::Mat img);

void bilinearInterpolation(); 

void createGuassKernel(float std_dev, int radius); 

void create1DDerivateGaussianKernel(float sigma, int radius);

