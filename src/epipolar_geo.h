#ifndef EPIPOLAR_GEO_H
#define EPIPOLAR_GEO_H

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/eigen.hpp>
#include <Eigen/Dense>
#include <vector>

// 8-points version of the algorithm.
cv::Mat estimateFundamentalMatrix8Pts(const std::vector<cv::Vec2d> x1, const std::vector<cv::Vec2d> x2);

//generic RANSAC
cv::Mat Ransac();

cv::Mat estimateEssentialMatrix();

cv::Matx33f standarizeCoords(const std::vector<cv::Vec2f> points);



#endif //EPIPOLAR_GEO_H
