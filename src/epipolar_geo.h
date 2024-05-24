#ifndef EPIPOLAR_GEO_H
#define EPIPOLAR_GEO_H

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/eigen.hpp>
#include <Eigen/Dense>
#include <vector>
#include <random>

// 8-points version of the algorithm.
Eigen::Matrix3f estimateFundamentalMatrix8Pts(const std::vector<cv::Vec2f>& x1, const std::vector<cv::Vec2f>& x2);

// 7-points version fo the algorithm
Eigen::Matrix3f estimateFundamentalMatrix7Pts(const std::vector<cv::Vec2f>& x1, const std::vector<cv::Vec2f>& x2);

// Ransac inlier detector
std::vector<std::pair<Eigen::Vector2f, Eigen::Vector2d>> inlierDetector(std::vector<cv::Vec2f>& points1, std::vector<cv::Vec2f>& points2, int iters, int maxEpipolarDist);

cv::Mat estimateEssentialMatrix();

Eigen::Matrix3f standarizeCoords(const std::vector<Eigen::Vector2f>& points);

std::vector<int> generateRandomNumbers(int min, int max, size_t count);

#endif //EPIPOLAR_GEO_H
