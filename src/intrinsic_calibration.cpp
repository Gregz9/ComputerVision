#include "intrinsic_calibration.h"

#include "opencv2/core/eigen.hpp"

IntrinsicCalibration::IntrinsicCalibration(cv::Matx33d K, cv::Vec5d dist_coeffs, cv::Size2i img_size)
    : K_cv_{K}
    , dist_coeffs_{dist_coeffs}
    , img_size_(img_size)
{
  cv::cv2eigen(K_cv_, K_);
}


const cv::Matx33d IntrinsicCalibration::K_cv() const
{
  return K_cv_;
}


const Eigen::Matrix3d IntrinsicCalibration::K() const
{
  return K_;
}

const cv::Vec5d IntrinsicCalibration::distCoeffs() const
{
  return dist_coeffs_;
}

const cv::Size2i IntrinsicCalibration::img_size() const
{
  return img_size_;
}
