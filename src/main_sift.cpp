#include "opencv2/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include <chrono>
#include "sift.h"
using namespace cv;
using namespace std;

int main(int argc, char** argv) {
    cv::Mat img = cv::imread("../IMG_2781.jpg");
    cv::Mat img2 = cv::imread("../IMG_2782.jpg");
    cv::resize(img, img, cv::Size(600, 600), cv::INTER_CUBIC);
    cv::resize(img2, img2, cv::Size(600, 600), cv::INTER_CUBIC);
    auto start = std::chrono::high_resolution_clock::now();
    keypoints kPoints1 = detect_keypoints(img, LAMB_DESC, LAMB_ORI);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    std::cout << "Elapsed time: " << elapsed_seconds.count() << " seconds" << std::endl;
    keypoints kPoints2 = detect_keypoints(img2, LAMB_DESC, LAMB_ORI);

    matches kMatches = match_keypoints(kPoints1, kPoints2);
    simplifiedMatches sim_matches = simplifyMatches(kMatches);
    std::vector<cv::Vec2f> keypoints1 = splitMatches(sim_matches, 0);
    std::vector<cv::Vec2f> keypoints2 = splitMatches(sim_matches, 1);
    drawMatchesKey(img, img2, kMatches);

    Ptr<cv::SIFT> detector = SIFT::create();

// Detect keypoints
    //cv::Mat img = cv::imread("../IMG_2781.jpg");
    std::vector<cv::KeyPoint> keypoints;
    auto start2 = std::chrono::high_resolution_clock::now();
    detector->detect(img, keypoints);
    auto end2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_seconds2 = end2 - start2;
    std::cout << "Elapsed time: " << elapsed_seconds2.count() << " seconds" << std::endl;

// Draw keypoints on the image
    cv::Mat img_keypoints;
    drawKeypoints(img, keypoints, img_keypoints, Scalar::all(-1), DrawMatchesFlags::DEFAULT);

// Print the number of detected keypoints
    cout << "Number of keypoints detected: " << keypoints.size() << endl;

// Display the image with keypoints
    imshow("Keypoints", img_keypoints);
    waitKey(0);
    return 0;
}