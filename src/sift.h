#ifndef MY_SIFT_H
#define MY_SIFT_H

// #include "opencv2/core.hpp"
#include <vector>
#include <opencv2/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/opencv.hpp>
#include <stdlib.h>
#include <cmath>
#include <limits>

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

// Values for matching
constexpr double REL_THR = 0.6;
constexpr int ABS_THR = 300; // 250 to 300
constexpr double MAX_DIST = std::numeric_limits<double>::infinity();

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
typedef std::vector<std::pair<Keypoint, Keypoint>> matches;

void drawKeypoints(cv::Mat& image, keypoints points);

void drawMatchesKey(const cv::Mat& img1, const cv::Mat& img2, const matches& key_matches);

void displayPyramid(const Pyramid& pyramid, int stride);

Pyramid computeGaussianPyramid(cv::Mat img);

Pyramid computeDoGPyramid(const Pyramid& pyramid);

Pyramid computeGradientImages(const Pyramid& scale_space);

keypoints locateExtrema(const Pyramid& dog, double C_dog = DOG_THR, double C_edge = EDGE_THR);

bool keypointRefinement(const Pyramid& DoG, Keypoint& k);

cv::Vec3d quadraticInterpolation(const Pyramid& DoG, Keypoint& k);

bool checkIfPointOnEdge(const Pyramid& Dog, const Keypoint& k, double C_edge);

std::vector<double> computeReferenceOrientation(Keypoint& k_points, const Pyramid& scaleSpaceGrads, double lamb_ori, double lamb_desc);

std::vector<double> buildKeypointDescriptor(Keypoint& k, const double ori, const Pyramid& scaleSpaceGrads, double lamb_descr, double* w_hist);

keypoints detect_keypoints(cv::Mat img, double lamd_descr, double lamb_ori);

matches match_keypoints(const keypoints& keypoints_img1, const keypoints& keypoints_img2, double rThr=REL_THR, int dThr=ABS_THR);

double computeEuclidenDist(const Keypoint& k1, const Keypoint& k2);

#endif /* MY_SIFT_H */