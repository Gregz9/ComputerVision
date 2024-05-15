#include "opencv2/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/opencv.hpp"
#include "cstdlib"
#include <vector>
#include "sift.h"
#include <opencv2/xfeatures2d.hpp>
using namespace cv;
using namespace cv::xfeatures2d;
using namespace std;

cv::Mat computeAbsolute(const cv::Mat& inputImage) {
    // Create a new matrix to store the absolute values
    cv::Mat absoluteImage(inputImage.size(), inputImage.type());

    // Iterate over each pixel and compute the absolute value
    for (int i = 0; i < inputImage.rows; ++i) {
        for (int j = 0; j < inputImage.cols; ++j) {
            // Compute the absolute value of the pixel value
            absoluteImage.at<double>(i, j) = std::abs(inputImage.at<double>(i, j));
        }
    }

    return absoluteImage;
}
void drawPoints(cv::Mat& image, keypoints points) {
    // Loop through each point and draw it on the image
    for (const auto& point : points) {
        cv::circle(image, cv::Point(point.x, point.y), 3, cv::Scalar(0, 0, 255), cv::FILLED);
    }

    // Display the image
    cv::imshow("Image with Points", image);
    cv::waitKey(0);
}

int main(int argc, char** argv)
{
    cv::Mat img = cv::imread("../IMG_2781.jpg");
    cv::resize(img,img, cv::Size(700,700), 0,0,cv::INTER_CUBIC);
    cv::Mat gray_image;
    cv::cvtColor(img, gray_image, cv::COLOR_BGR2GRAY);


    std::cout << img.size << std::endl;
    gray_image.convertTo(gray_image, CV_64F);
    cv::normalize(gray_image, gray_image, 0, 1, cv::NORM_MINMAX, CV_64F);

    Pyramid pyramid = computeGaussianPyramid(gray_image);

    Pyramid DoG = computeDoGPyramid(pyramid);
    keypoints k_points = locateExtrema(DoG);
    drawPoints(gray_image, k_points);

    //keypoints refined_Ks = keypointRefinement(DoG, k_points);

    std::cout << k_points.size() << std::endl;
    //std::cout << refined_Ks.size() << std::endl;

    std::cout << pyramid.imgs.size() << std::endl;
    std::cout << pyramid.num_oct << std::endl;

    //test convolution:

    vector<float> elements = {1, 2, 4, 3};

    // Convert vector to cv::Mat
    Mat array(1, elements.size(), CV_32F);
    for (int i = 0; i < elements.size(); ++i) {
        array.at<float>(0, i) = elements[i];
    }

    cv::Mat kernel = Mat(1, 3, CV_32F, 1.0/3);
    Mat result;
    filter2D(array, result, -1, kernel, cv::Point(-1, -1), 0, BORDER_WRAP);
    for(int i = 0; i < 3; ++i) {
        std::cout << result.at<float>(i) << ", ";
    }
    std::cout << "\n";

    cv::Mat image1 = (pyramid.imgs.at(0));
    cv::Mat image2 = (pyramid.imgs.at(1));
    cv::Mat image3 = (pyramid.imgs.at(3));
    cv::Mat image4 = (pyramid.imgs.at(10));
    cv::Mat image5 = (pyramid.imgs.at(20));
    cv::Mat image6 = (pyramid.imgs.at(30));


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

    /*cv::Mat some_nums(3,3, CV_64FC1);
    cv::randu(some_nums, 0, 255);

    std::cout << some_nums.at<double>(2,0) << std::endl;
    std::cout << some_nums.at<double>(6) << std::endl;*/

    return 0;
}