#pragma once

// #include "opencv2/core.hpp"
#include <vector>
#include <opencv2/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/opencv.hpp>
#include <stdlib.h>
#include <cmath>

constexpr int NUM_OCT = 8;
constexpr int NUM_SCL_PER_OCT = 3;
constexpr double MIN_SIGMA = 0.8; // Minimal blur level
constexpr double INP_SIGMA = 0.5; // Assumed blur level of input image
constexpr double IN_PX_DST = 1.0;
constexpr double PX_DST_MIN = 0.5;

constexpr int MAX_INTER_ITERS = 5;
constexpr double MAX_REF_THR = 0.5;
constexpr double DOG_THR = 0.015;
constexpr double EDGE_THR = 10.;

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
    double sigma = 0.f; //blur level
    double omega = 0.f; //intensity of the extremum
};

typedef std::vector<KeyPoint> keypoints;

Pyramid computeGaussianPyramid(const cv::Mat img);

Pyramid computeDoGPyramid(const Pyramid& pyramid);

Pyramid computeGradientImages(Pyramid scale_space);

keypoints locateExtrema(const Pyramid dog, double C_dog = DOG_THR, double C_edge = EDGE_THR);

bool keypointRefinement(const Pyramid& DoG, KeyPoint& k);

cv::Vec3d quadraticInterpolation(const Pyramid& DoG, KeyPoint& k);

bool checkIfPointOnEdge(Pyramid Dog, KeyPoint k, double C_edge);

