/*
// Create a SIFT detector
Ptr<cv::SIFT> detector = SIFT::create();

// Detect keypoints
std::vector<cv::KeyPoint> keypoints;
detector->detect(gray_image, keypoints);

// Draw keypoints on the image
cv::Mat img_keypoints;
drawKeypoints(img, keypoints, img_keypoints, Scalar::all(-1), DrawMatchesFlags::DEFAULT);

// Print the number of detected keypoints
cout << "Number of keypoints detected: " << keypoints.size() << endl;

// Display the image with keypoints
imshow("Keypoints", img_keypoints);
waitKey(0);*/
//img = rgb_to_grayscale(img);
//cv::resize(img, img, cv::Size(200,200));
// Remember that the image has to be converted to double values