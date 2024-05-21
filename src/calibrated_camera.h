#pragma once

#include "frame.h"
#include "intrinsic_calibration.h"

class CalibratedCamera
{
public:
  virtual ~CalibratedCamera() = default;
  virtual const IntrinsicCalibration& calibration() = 0;
  virtual cv::Mat captureImage()  = 0;
};
