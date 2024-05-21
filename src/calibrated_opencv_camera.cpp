#include "calibrated_opencv_camera.h"

IntrinsicCalibration setupCalibration()
{
  // Set K matrix.
  cv::Matx33d K
      {6.6051081297156020e+02, 0., 3.1810845757653777e+02,
       0., 6.6051081297156020e+02, 2.3995332228230293e+02,
       0., 0., 1.};

  // Set distortion coefficients [k1, k2, 0, 0, k3].
  cv::Vec5d dist_coeffs{
      0., 2.2202255011309072e-01, 0., 0., -5.0348071005413975e-01};

  // Set image size.
  cv::Size2i img_size(640, 480);

  return {K, dist_coeffs, img_size};
}

CalibratedOpencvCamera::CalibratedOpencvCamera(int camera_id)
  : cap_(camera_id)
  , calibration_(setupCalibration())
{
  cap_.set(cv::CAP_PROP_FRAME_WIDTH, calibration_.img_size().width);
  cap_.set(cv::CAP_PROP_FRAME_HEIGHT, calibration_.img_size().height);

  if (!cap_.isOpened())
  {
    throw std::runtime_error("Could not open camera " + std::to_string(camera_id));
  }

}

const IntrinsicCalibration& CalibratedOpencvCamera::calibration()
{
  return calibration_;
}

cv::Mat CalibratedOpencvCamera::captureImage()
{
  cv::Mat image;
  cap_ >> image;

  if (image.empty())
  {
    throw std::runtime_error("Could not capture from camera");
  }

  return image;
}
