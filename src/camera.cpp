#include "calibrated_opencv_camera.h"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#ifdef TEK5030_REALSENSE
#include "calibrated_realsense_camera.h"
#endif
#include <iostream>

int main(int argc, char* argv[]) try {
        constexpr int camera_id = 0;
        auto camera = std::make_shared<CalibratedRealSenseCamera>();

        const std::string window_name{"frame"};
        cv::namedWindow(window_name, cv::WINDOW_NORMAL);

        while (true) {
            cv::Mat distorted_image = camera->captureImage();

            if (distorted_image.empty())
                break;

            cv::imshow(window_name, distorted_image);
            auto key = cv::waitKey(1);
            if (key >= 0) { break; }
        }
        return EXIT_SUCCESS;
    } catch (const std::exception& e)
    {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }