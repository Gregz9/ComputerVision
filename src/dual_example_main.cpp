#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "tek5030/realsense_stereo.h"
#include <iostream>

int main() try
{
    using namespace tek5030;
    using namespace tek5030::RealSense;

    StereoCamera cam(StereoCamera::CaptureMode::UNRECTIFIED);

    std::cout << "Connected to RealSense camera: " << std::endl;
    std::cout << "Resolution: " << cam.getResolution(StereoCamera::CameraStream::LEFT) << std::endl;
    std::cout << "Frame rate: " << cam.getFrameRate(StereoCamera::CameraStream::LEFT) << std::endl;
    std::cout << std::endl << "Press 'l' to toggle laser." << std::endl;

    const std::string window_name{"frame"};
    cv::namedWindow(window_name, cv::WINDOW_NORMAL);

    bool laser_on = false;

    while (true)
    {
        StampedStereoPair stereo_pair;
        cam >> stereo_pair;

        cv::Mat pair;
        cv::hconcat(stereo_pair.stereo_pair.left, stereo_pair.stereo_pair.right, pair);
        cv::cvtColor(pair, pair, cv::COLOR_GRAY2BGR);

        // Draw lines along the image rows.
        // For rectified images, these should coincide with the epipolar lines.
        for (int i=50; i < pair.rows; i += 50)
        {
            cv::line(pair, {0, i}, {pair.cols, i}, {0,0,65535});
        }

        cv::imshow(window_name, pair);

        auto key = cv::waitKey(1);
        if (key == static_cast<int>('l'))
        {
            laser_on = !laser_on;
            cam.setLaserMode(laser_on ? StereoCamera::LaserMode::ON : StereoCamera::LaserMode::OFF);
        }
        else if (key >= 0)
        { break; }
    }
    return EXIT_SUCCESS;
}
catch (const std::exception& e)
{
    std::cerr << e.what() << std::endl;
    return EXIT_FAILURE;
}