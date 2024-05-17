#include "opencv2/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "sift.h"
using namespace cv;
using namespace std;

int main(int argc, char** argv)
{
    cv::Mat img = cv::imread("../IMG_2781.jpg");
    cv::Mat img2 = cv::imread("../IMG_2782.jpg");
    keypoints kPoints1 = detect_keypoints(img, LAMB_DESC, LAMB_ORI);
    keypoints kPoints2 = detect_keypoints(img2, LAMB_DESC, LAMB_ORI);

    matches kMatches = match_keypoints(kPoints1, kPoints2);
    drawMatchesKey(img, img2, kMatches);
    return 0;
}