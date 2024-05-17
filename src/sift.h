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

// Histogram specific values
constexpr int N_BINS = 36;
constexpr int N_HISTS = 4;
constexpr int N_ORI = 8;
constexpr double LAMB_ORI = 1.5;
constexpr double LAMB_DESC = 6.;

struct Pyramid{
  int num_oct = NUM_OCT;
  int num_scales_per_oct;
  std::vector<cv::Mat> imgs{};
};

struct Keypoint {
    int m;
    int n;
    int octave;
    int scale;

    int x = 0;
    int y = 0;
    double sigma = 0.f; //blur level
    double omega = 0.f; //intensity of the extremum
    std::vector<double> ref_oris{}; // Reference orientations
    std::vector<double> descriptor;
};

typedef std::vector<Keypoint> keypoints;

Pyramid computeGaussianPyramid(cv::Mat img);

Pyramid computeDoGPyramid(const Pyramid& pyramid);

Pyramid computeGradientImages(const Pyramid& scale_space);

keypoints locateExtrema(const Pyramid& dog, double C_dog = DOG_THR, double C_edge = EDGE_THR);

bool keypointRefinement(const Pyramid& DoG, Keypoint& k);

cv::Vec3d quadraticInterpolation(const Pyramid& DoG, Keypoint& k);

bool checkIfPointOnEdge(const Pyramid& Dog, const Keypoint& k, double C_edge);

std::vector<double> computeReferenceOrientation(Keypoint& k_points, const Pyramid& scaleSpaceGrads, double lamb_ori, double lamb_desc);

std::vector<double> buildKeypointDescriptor(Keypoint& k, const double ori, const Pyramid& scaleSpaceGrads, double lamb_descr, double* w_hist);

keypoints detect(cv::Mat img);