#ifndef SIFT_H
#define SIFT_H

#include "opencv2/core.hpp"

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

void computeDogThr(); 

void gaussianPyramid(cv::Mat img); 

void bilinearInterpolation(); 

void createGuassKernel(float std_dev, int radius); 

void create1DDerivateGaussianKernel(float sigma, int radius);

#endif /* SIFT_H */
