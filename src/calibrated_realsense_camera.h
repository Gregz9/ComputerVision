#pragma once

#include "tek5030/realsense_single.h"
#include "calibrated_camera.h"

class CalibratedRealSenseCamera : public CalibratedCamera
{
public:
  explicit CalibratedRealSenseCamera(
    const tek5030::RealSense::SingleStreamCamera::CameraStream& active_stream = tek5030::RealSense::SingleStreamCamera::CameraStream::LEFT
  );

  const IntrinsicCalibration& calibration() override;
  cv::Mat captureImage() override;

private:
  tek5030::RealSense::SingleStreamCamera cam_;
  IntrinsicCalibration calibration_;
};
