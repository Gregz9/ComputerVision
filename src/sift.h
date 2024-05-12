#pragma once

// #include "opencv2/core.hpp"
#include <vector>
#include <opencv2/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/opencv.hpp>
#include <stdlib.h>

constexpr int NUM_OCT = 8;
constexpr int NUM_SCL_PER_OCT = 3; 
constexpr float MIN_SIGMA = 0.8; // Minimal blur level
constexpr float INP_SIGMA = 0.5; // Assumed blur level of input image
constexpr float IN_PX_DST = 1.0; 
constexpr float PX_DST_MIN = 0.5;

constexpr int MAX_ITERS_REF = 5;
constexpr float MAX_REF_THR = 0.5; 
constexpr float DOG_THR = 0.015f;
constexpr float EDGE_THR = 10.f;

struct Pyramid{
  int num_oct = NUM_OCT;
  int num_scales_per_oct;
  std::vector<cv::Mat> imgs{}; 
};

struct KeyPoint {
    int m;
    int n;
    int octave;
    int scale;

    int x = 0;
    int y = 0;
    float sigma = 0.f; //blur level
    float omega = 0.f; //intensity of the extremum
};
void computeDogThr();

typedef std::vector<KeyPoint> keypoints;

Pyramid computeGaussianPyramid(const cv::Mat img);

Pyramid computeDoGPyramid(const Pyramid pyramid);

keypoints locateExtrema(const Pyramid dog, float C_dog = DOG_THR, float C_edge = EDGE_THR);

