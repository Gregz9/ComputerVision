#include "frame.h"

#include "opencv2/calib3d.hpp"
#include "opencv2/core/eigen.hpp"
#include "opencv2/imgproc.hpp"

Frame::Frame(const IntrinsicCalibration& calibration,
             const cv::Mat& distorted_image,
             cv::Ptr<cv::FeatureDetector> detector,
             cv::Ptr<cv::DescriptorExtractor> desc_extractor,
             int max_num_keypoints)
    : K_{calibration.K()}
    , K_inv_{calibration.K().inverse()}
{
  // Undistort and store resulting image.
  cv::undistort(distorted_image, image_, calibration.K_cv(), calibration.distCoeffs());

  // Detect and store features with descriptors.
  cv::Mat gray_img = image_;
  if (gray_img.channels() > 1)
  {
    cv::cvtColor(gray_img, gray_img, cv::COLOR_BGR2GRAY);
  }
  detector->detect(gray_img, keypoints_);
  cv::KeyPointsFilter::retainBest(keypoints_, max_num_keypoints);
  desc_extractor->compute(gray_img, keypoints_, descriptors_);
}


cv::Mat Frame::image() const
{
  return image_.clone();
}


const Eigen::Matrix3d Frame::K() const
{
  return K_;
}


const std::vector<cv::KeyPoint> Frame::keypoints() const
{
  return keypoints_;
}


cv::Mat Frame::descriptors() const
{
  return descriptors_.clone();
}

void Frame::setPose(const Sophus::SE3d& pose_W_C)
{
  pose_W_C_ = pose_W_C;
  has_pose_ = true;
}

const Sophus::SE3d Frame::pose_W_C() const
{
  return pose_W_C_;
}

bool Frame::hasPose() const
{
  return has_pose_;
}

Eigen::Vector2d Frame::toNormalized(const Eigen::Vector2d& u) const
{
  return (K_inv_ * u.homogeneous()).hnormalized();
}
