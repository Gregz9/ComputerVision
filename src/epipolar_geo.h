#ifndef EPIPOLAR_GEO_H
#define EPIPOLAR_GEO_H

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/mat.hpp>
#include <Eigen/Dense>
#include <vector>

// 8-points version of the algorithm.
Eigen::Matrix3f estimateFundamentalMatrix8Pts(const std::vector<cv::Vec2f>& x1, const std::vector<cv::Vec2f>& x2);

// 7-points version fo the algorithm
Eigen::Matrix3f estimateFundamentalMatrix7Pts(const std::vector<cv::Vec2f>& x1, const std::vector<cv::Vec2f>& x2);

//generic RANSAC
cv::Mat Ransac();

cv::Mat estimateEssentialMatrix();

Eigen::Matrix3f standarizeCoords(const std::vector<Eigen::Vector2f>& points);



#endif //EPIPOLAR_GEO_H
