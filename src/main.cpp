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
    img.convertTo(img, CV_32F);
    //cv::normalize(img, img, 0, 1, cv::NORM_MINMAX);


    Pyramid pyramid = computeGaussianPyramid(img);
    Pyramid DoG = computeDoGPyramid(pyramid);

    std::cout << pyramid.imgs.size() << std::endl;
    std::cout << pyramid.num_oct << std::endl;

    cv::Mat image1 = computeAbsolute(DoG.imgs.at(0));
    cv::Mat image2 = computeAbsolute(DoG.imgs.at(1));
    cv::Mat image3 = computeAbsolute(DoG.imgs.at(15));
    cv::Mat image4 = computeAbsolute(DoG.imgs.at(20));
    cv::Mat image5 = computeAbsolute(DoG.imgs.at(25));
    cv::Mat image6 = computeAbsolute(DoG.imgs.at(30));
    for (size_t i = 0; i < DoG.imgs.size(); ++i) {
        cv::Mat current_img = DoG.imgs[i];
        cv::Scalar mean_val = cv::mean(current_img);
        std::cout << "Mean pixel value of image " << i << ": " << mean_val[0] << std::endl;
    }
    cv::normalize(image1, image1, 0, 1, cv::NORM_MINMAX);
    cv::normalize(image2, image2, 0, 1, cv::NORM_MINMAX);
    cv::normalize(image3, image3, 0, 1, cv::NORM_MINMAX);
    cv::normalize(image4, image4, 0, 1, cv::NORM_MINMAX);
    cv::normalize(image5, image5, 0, 1, cv::NORM_MINMAX);
    cv::normalize(image6, image6, 0, 1, cv::NORM_MINMAX);

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