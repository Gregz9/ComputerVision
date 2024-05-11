#include "opencv2/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/opencv.hpp"
#include "cstdlib"
#include <vector>
#include "sift.h"

int main(int argc, char** argv)
{
    cv::Mat img = cv::imread("../IMG_2781.jpg", cv::IMREAD_GRAYSCALE);
    cv::resize(img, img, cv::Size(200,200));
    cv::Mat grayMat, colorMat;

    std::cout << img.size << std::endl;
    GaussianPyramid pyramid = computeGaussianPyramid(img);

    std::cout << pyramid.imgs.size() << std::endl;

    cv::Mat image1 = img;
    cv::Mat image2 = pyramid.imgs.at(11);
    cv::Mat image3 = pyramid.imgs.at(17);
    cv::Mat image4 = pyramid.imgs.at(23);
    cv::Mat image5 = pyramid.imgs.at(29);
    //cv::Mat image6 = pyramid.imgs.at(29);

    cv::normalize(image2, image2, 0, 255, cv::NORM_MINMAX);
    cv::normalize(image3, image3, 0, 255, cv::NORM_MINMAX);
    cv::normalize(image4, image4, 0, 255, cv::NORM_MINMAX);
    cv::normalize(image5, image5, 0, 255, cv::NORM_MINMAX);
    //cv::normalize(image6, image6, 0, 255, cv::NORM_MINMAX);

    // Check if images are loaded successfully
    if (image1.empty() || image2.empty() || image3.empty() || image4.empty() || image5.empty()) {
        std::cerr << "Error: Unable to load images!" << std::endl;
        return -1;
    }

    // Define the common size for all images
    cv::Size commonSize(400, 400);

    // Resize images to the common size
    cv::resize(image1, image1, commonSize);
    cv::resize(image2, image2, commonSize);
    cv::resize(image3, image3, commonSize);
    cv::resize(image4, image4, commonSize);
    cv::resize(image5, image5, commonSize);
    //cv::resize(image6, image6, commonSize);

    // Create a window to display the grid of images
    cv::namedWindow("Grid of Images", cv::WINDOW_NORMAL);

    // Create a single image to display the grid of images
    cv::Mat gridImage(commonSize.height * 2 + 10, commonSize.width * 3 + 20, image1.type());

    // Copy images to the gridImage
    image1.copyTo(gridImage(cv::Rect(0, 0, commonSize.width, commonSize.height)));
    image2.copyTo(gridImage(cv::Rect(commonSize.width + 5, 0, commonSize.width, commonSize.height)));
    image3.copyTo(gridImage(cv::Rect(commonSize.width * 2 + 10, 0, commonSize.width, commonSize.height)));
    image4.copyTo(gridImage(cv::Rect(0, commonSize.height + 5, commonSize.width, commonSize.height)));
    image5.copyTo(gridImage(cv::Rect(commonSize.width + 5, commonSize.height + 5, commonSize.width, commonSize.height)));
    //image6.copyTo(gridImage(cv::Rect(commonSize.width * 2 + 10, commonSize.height + 5, commonSize.width, commonSize.height)));
    // Display the grid of images
    cv::imshow("Grid of Images", gridImage);

    // Wait for a key press
    cv::waitKey(0);


    return 0;
}