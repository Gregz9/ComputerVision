#include "sift.h"

void gaussianPyramid(cv::Mat img){

  // First octave
  img = cv::resize(img.rows*2, img.cols*2, cv::INTER_LINEAR);
  // Computing the standard deviation of the gaussian kernel
}
