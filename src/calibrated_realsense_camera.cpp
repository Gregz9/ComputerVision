#include "calibrated_realsense_camera.h"

CalibratedRealSenseCamera::CalibratedRealSenseCamera(
  const tek5030::RealSense::SingleStreamCamera::CameraStream& active_stream)
  : cam_(active_stream)
  , calibration_(cam_.K(), cam_.distortion(), cam_.getResolution())
{
  cam_.setLaserMode(tek5030::RealSense::SingleStreamCamera::LaserMode::OFF);
}

const IntrinsicCalibration& CalibratedRealSenseCamera::calibration()
{
  return calibration_;
}

cv::Mat CalibratedRealSenseCamera::captureImage()
{
  return cam_.getFrame();
}
