#include "opencv2/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/opencv.hpp"
#include "cstdlib"
#include <vector>
#include "sift.h"

cv::Mat computeAbsolute(const cv::Mat& inputImage) {
    // Create a new matrix to store the absolute values
    cv::Mat absoluteImage(inputImage.size(), inputImage.type());

    // Iterate over each pixel and compute the absolute value
    for (int i = 0; i < inputImage.rows; ++i) {
        for (int j = 0; j < inputImage.cols; ++j) {
            // Compute the absolute value of the pixel value
            absoluteImage.at<float>(i, j) = std::abs(inputImage.at<float>(i, j));
        }
    }

    return absoluteImage;
}


int main(int argc, char** argv)
{
    cv::Mat img = cv::imread("../book_in_scene.jpg", cv::IMREAD_GRAYSCALE);
    //cv::resize(img, img, cv::Size(200,200));
    cv::Mat grayMat, colorMat;
    // Remember that the image has to be converted to float values

    img.convertTo(img, CV_32F);
    img /= 255.;


    Pyramid pyramid = computeGaussianPyramid(img);
    std::cout << "\n" << std::endl;
    Pyramid DoG = computeDoGPyramid(pyramid);
    keypoints k_points = locateExtrema(DoG);

    std::cout << k_points.size() << std::endl;

    std::cout << pyramid.imgs.size() << std::endl;
    std::cout << pyramid.num_oct << std::endl;

    cv::Mat image1 = (DoG.imgs.at(0));
    cv::Mat image2 = (DoG.imgs.at(7));
    cv::Mat image3 = (DoG.imgs.at(14));
    cv::Mat image4 = (DoG.imgs.at(21));
    cv::Mat image5 = (DoG.imgs.at(28));
    cv::Mat image6 = (DoG.imgs.at(35));


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
    cv::resize(image6, image6, commonSize);

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
    image6.copyTo(gridImage(cv::Rect(commonSize.width * 2 + 10, commonSize.height + 5, commonSize.width, commonSize.height)));
    // Display the grid of images
    cv::imshow("Grid of Images", gridImage);

    // Wait for a key press
    cv::waitKey(0);

    /*cv::Mat some_nums(3,3, CV_32FC1);
    cv::randu(some_nums, 0, 255);

    std::cout << some_nums.at<float>(2,0) << std::endl;
    std::cout << some_nums.at<float>(6) << std::endl;*/

    return 0;
}