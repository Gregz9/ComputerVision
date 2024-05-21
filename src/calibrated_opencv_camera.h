#pragma once

#include "calibrated_camera.h"
#include "opencv2/videoio.hpp"

class CalibratedOpencvCamera : public CalibratedCamera
{
public:
  explicit CalibratedOpencvCamera(int camera_id);

  const IntrinsicCalibration& calibration() override;
  cv::Mat captureImage() override;

private:
  cv::VideoCapture cap_;
  IntrinsicCalibration calibration_;
};
