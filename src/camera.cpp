#include "calibrated_opencv_camera.h"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#ifdef TEK5030_REALSENSE
#include "calibrated_realsense_camera.h"
#endif
#include <iostream>
#include "sift.h"

int main(int argc, char* argv[]) try {
        constexpr int camera_id = 0;
        auto camera = std::make_shared<CalibratedRealSenseCamera>();

        const std::string window_name{"frame"};
        cv::namedWindow(window_name, cv::WINDOW_NORMAL);
        cv::Mat img1, img2;
        bool first_image_taken = false;
        while (true) {
            cv::Mat distorted_image = camera->captureImage();

            if (distorted_image.empty())
                break;

            cv::imshow(window_name, distorted_image);
            auto key = cv::waitKey(1);
            //if (key >= 0) { break; }
            if (key == ' ') {
                if (!first_image_taken) {
                    img1 = distorted_image.clone();
                    first_image_taken = true;
                    std::cout << "First image captured.\n";
                } else {
                    img2 = distorted_image.clone();
                    std::cout << "Second image captured.\n";
                    break;
                }
            }
        }
        cv::resize(img1, img1, cv::Size(600, 600), cv::INTER_CUBIC);
        cv::resize(img2, img2, cv::Size(600, 600), cv::INTER_CUBIC);
        keypoints kPoints1 = detect_keypoints(img1, LAMB_DESC, LAMB_ORI);
        keypoints kPoints2 = detect_keypoints(img2, LAMB_DESC, LAMB_ORI);

        matches kMatches = match_keypoints(kPoints1, kPoints2);
        //simplifiedMatches sim_matches = simplifyMatches(kMatches);
        //std::vector<cv::Vec2f> keypoints1 = splitMatches(sim_matches, 0);
        //std::vector<cv::Vec2f> keypoints2 = splitMatches(sim_matches, 1);
        drawMatchesKey(img1, img2, kMatches);

        return EXIT_SUCCESS;
    } catch (const std::exception& e)
    {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }