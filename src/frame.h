#pragma once

#include "intrinsic_calibration.h"

#include "Eigen/Core"
#include "opencv2/features2d.hpp"
#include "sophus/se3.hpp"
#include <memory>

class Frame
{
public:
  /// Undistorts, extracts keypoints, descriptors,
  Frame(const IntrinsicCalibration& calibration,
        const cv::Mat& distorted_image,
        cv::Ptr<cv::FeatureDetector> detector,
        cv::Ptr<cv::DescriptorExtractor> desc_extractor,
        int max_num_keypoints = 1000);

  cv::Mat image() const;
  const Eigen::Matrix3d K() const;
  const std::vector<cv::KeyPoint> keypoints() const;
  cv::Mat descriptors() const;
  void setPose(const Sophus::SE3d& pose_W_C);
  const Sophus::SE3d pose_W_C() const;
  bool hasPose() const;
  Eigen::Vector2d toNormalized(const Eigen::Vector2d& u) const;

private:
  Eigen::Matrix3d K_;
  Eigen::Matrix3d K_inv_;
  cv::Mat image_;
  std::vector<cv::KeyPoint> keypoints_;
  cv::Mat descriptors_;
  Sophus::SE3d pose_W_C_;
  bool has_pose_ = false;
};
