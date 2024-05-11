std::vector<cv::Mat> imgs {};
imgs.push_back(cv::imread("../IMG_2781.jpg"));
imgs.push_back(cv::imread("../IMG_2782.jpg"));

cv::Mat image1 = imgs.at(0);
cv::Mat image2 = imgs.at(1);

// Check if images are loaded successfully
if (image1.empty() || image2.empty()) {
std::cerr << "Error: Unable to load images!" << std::endl;
return -1;
}

cv::resize(image1, image1, cv::Size(600, 600), cv::INTER_CUBIC);
cv::resize(image2, image2, cv::Size(600, 600), cv::INTER_CUBIC);
// Create a window to display the images
cv::namedWindow("Side by Side", cv::WINDOW_NORMAL);

// Create a single image to display both images side by side
cv::Mat sideBySide(image1.rows, image1.cols + image2.cols, image1.type());

// Copy first image to the left half of sideBySide
image1.copyTo(sideBySide(cv::Rect(0, 0, image1.cols, image1.rows)));

// Copy second image to the right half of sideBySide
image2.copyTo(sideBySide(cv::Rect(image1.cols, 0, image2.cols, image2.rows)));

// Display the combined image
cv::imshow("Side by Side", sideBySide);

// Wait for a key press
cv::waitKey(0);
