#include "sift.h"
#include "filters.h"
#include <fstream>

void drawKeypoints(cv::Mat& image, keypoints points) {
    // Loop through each point and draw it on the image
    for (const auto& point : points) {
        cv::circle(image, cv::Point(point.x, point.y), 3, cv::Scalar(0, 0, 255), 1); // Change cv::FILLED to 1
    }

    // Display the image
    cv::imshow("Image with Points", image);
    cv::waitKey(0);
}

void displayPyramid(const Pyramid& pyramid, int stride) {

    cv::Mat image1 = (pyramid.imgs.at(0));
    cv::Mat image2 = (pyramid.imgs.at(stride*1));
    cv::Mat image3 = (pyramid.imgs.at(stride*2));
    cv::Mat image4 = (pyramid.imgs.at(stride*3));
    cv::Mat image5 = (pyramid.imgs.at(stride*4));
    cv::Mat image6 = (pyramid.imgs.at(stride*5));

    if (image1.empty() || image2.empty() || image3.empty() || image4.empty() || image5.empty()) {
        std::cerr << "Error: Unable to load images!" << std::endl;
        exit(1);
    }

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
}
Pyramid computeGaussianPyramid(cv::Mat img){

    cv::resize(img, img, cv::Size(img.cols*2, img.rows*2),0,0, cv::INTER_LINEAR);
    // Computing the standard deviation of the gaussian kernel
    // For the very first image, init_sigma is equivalent to:

    double std_dev = std::sqrt((MIN_SIGMA*MIN_SIGMA) - (INP_SIGMA*INP_SIGMA));
    std_dev /= PX_DST_MIN;

    //cv::Mat kernel = create1DGaussKernel(std_dev);
    //cv::sepFilter2D(img, img, CV_64F, kernel, kernel);
    cv::GaussianBlur(img, img, cv::Size(), std_dev, std_dev);

    int num_scales = NUM_SCL_PER_OCT + 3;
    double min_coef = MIN_SIGMA / PX_DST_MIN;
    auto k = static_cast<double>(std::pow(2, 1.0/NUM_SCL_PER_OCT));

    std::vector<double> std_devs {min_coef};
    Pyramid pyramid;
    pyramid.num_scales_per_oct = num_scales;

    // Computing standard deviations
    for (int i=1; i<num_scales; ++i){
        auto prev_sigma = static_cast<double>(min_coef * std::pow(k, i-1));
        double sigma = k * prev_sigma;
        std_devs.push_back(std::sqrt(sigma*sigma - prev_sigma*prev_sigma));
    }

    pyramid.imgs.reserve(num_scales*NUM_OCT);
    for (int i = 0; i < NUM_OCT; i++) {
        pyramid.imgs.push_back(img);

        for (int j = 1; j < static_cast<int>(std_devs.size()); j++) {
            // Old solution
            /*cv::Mat scale_img = pyramid.imgs.at(i*num_scales).clone();
            cv::Mat blur_kernel = create1DGaussKernel(std_devs[j]);
            cv::sepFilter2D(scale_img, scale_img, CV_64F, blur_kernel, blur_kernel);
            pyramid.imgs.push_back(scale_img.clone());*/

            cv::Mat scale_img = pyramid.imgs.at(i * num_scales + j - 1);
            cv::Mat blur_result;
            cv::GaussianBlur(scale_img, blur_result, cv::Size(), std_devs[j], std_devs[j]);
            pyramid.imgs.push_back(blur_result);
        }
        cv::Mat next_img = pyramid.imgs.at(((i + 1) * num_scales) - NUM_SCL_PER_OCT);
        cv::resize(next_img, img, cv::Size(next_img.cols / 2, next_img.rows / 2), 0, 0, cv::INTER_LINEAR);
    }
    return pyramid;
}

Pyramid computeDoGPyramid(const Pyramid& gauss) {
    Pyramid dog = { gauss.num_oct, gauss.num_scales_per_oct - 1 };
    dog.imgs.reserve(dog.num_scales_per_oct * dog.num_oct);

    for (int o = 0; o < dog.num_oct; o++) {
        for (int s = 1; s < gauss.num_scales_per_oct; s++) { // Start from s = 0
            cv::Mat diff_img;
            cv::subtract(gauss.imgs[(o * gauss.num_scales_per_oct) + s],
                         gauss.imgs[(o * gauss.num_scales_per_oct) + s - 1], diff_img);
            dog.imgs.push_back(diff_img.clone());
        }
    }
    return dog;
}

Pyramid computeGradientImages(const Pyramid& scale_space) {

    Pyramid ScaleSpaceGradients = {
            scale_space.num_oct,
            scale_space.num_scales_per_oct,
            //std::vector<cv::Mat>(scale_space.imgs.size())
    };

    for(int o = 0; o < scale_space.num_oct; ++o) {
        for(int s = 0; s < scale_space.num_scales_per_oct; ++s) {
            cv::Mat curr_img = scale_space.imgs[(o*scale_space.num_scales_per_oct)+s];
            cv::Mat grad_img(curr_img.size(), CV_64FC2);
            for(int i = 1; i < curr_img.rows-1; ++i) {
                for(int j = 1; j < curr_img.cols-1; ++j) {
                    double gx = 0.5*(curr_img.at<double>(i+1, j) - curr_img.at<double>(i-1, j));
                    double gy = 0.5*(curr_img.at<double>(i, j+1) - curr_img.at<double>(i, j-1));
                    cv::Vec2d grads = {gx, gy};
                    grad_img.at<cv::Vec2d>(i,j) = grads;
                }
            }
            ScaleSpaceGradients.imgs.push_back(grad_img);
        }
    }
    return ScaleSpaceGradients;
}

keypoints locateExtrema(const Pyramid& dog, double C_dog, double C_edge) {
    // Impossible to know how many keypoint candidates there'll
    // be, hence cannot reserve a specific size for the vector.
    keypoints k_points{};
    for(int o = 0; o < dog.num_oct; ++o) {
        // DoG pyramid has already num_scales-1, hence only 1 subtracted
        for(int s=1; s < dog.num_scales_per_oct-1; ++s) {

            const cv::Mat& img_DoG = dog.imgs[((o*dog.num_scales_per_oct)+s)];
            const cv::Mat& prev_img_DoG = dog.imgs[((o*dog.num_scales_per_oct)+s-1)];
            const cv::Mat& next_img_DoG = dog.imgs[((o*dog.num_scales_per_oct)+s+1)];

            for(int i = 1; i < img_DoG.rows -1; ++i) {
                for(int j =1; j < img_DoG.cols -1; ++j) {
                    double current_pixel = img_DoG.at<double>(i,j);
                    // It's more efficient to check for the threshold criteria here, rather than later.
                    if(std::abs(current_pixel) < 0.8f*C_dog) {
                        continue;
                    }

                    bool max = true, min = true;

                    for(int x = -1; x<2; ++x) {
                        for (int y = -1; y < 2; ++y) {
                            auto neighbor_pixel_curr = img_DoG.at<double>(i + x, j + y);
                            auto neighbor_pixel_prev = prev_img_DoG.at<double>(i + x, j + y);
                            auto neighbor_pixel_next = next_img_DoG.at<double>(i + x, j + y);

                            if (current_pixel < neighbor_pixel_prev || current_pixel < neighbor_pixel_curr ||
                                current_pixel < neighbor_pixel_next) {
                                max = false;
                            }
                            if (current_pixel > neighbor_pixel_prev || current_pixel > neighbor_pixel_curr ||
                                current_pixel > neighbor_pixel_next) {
                                min = false;
                            }
                        }
                    }
                    if (max || min) {
                        Keypoint k = {i, j, o, s};
                        bool valid = keypointRefinement(dog, k);
                        if(valid) {
                            k_points.push_back(k);
                        }
                    }
                }
            }
        }
    }
    return k_points;
}

bool keypointRefinement(const Pyramid& DoG, Keypoint& k) {
        bool is_valid = false;

        for(int t = 0; t < MAX_INTER_ITERS; t++) {
            // Quadratic interpolation
            cv::Vec3d alfas = quadraticInterpolation(DoG, k);
            k.scale += cvRound(alfas[0]);
            k.m += cvRound(alfas[1]);
            k.n += cvRound(alfas[2]);
            if(k.scale >= DoG.num_scales_per_oct-1 || k.scale < 1)
                break;

            double quad_extremum = std::max({std::abs(alfas[0]), std::abs(alfas[1]), std::abs(alfas[2])});
            bool not_edge = checkIfPointOnEdge(DoG, k, EDGE_THR);
            bool valid_contr = std::abs(k.omega) >= DOG_THR;

            if (quad_extremum < 0.6 && valid_contr && not_edge) {
                // Continuous values of sigma and image coordinates. RMB! PX_DIST(o) = PX_DST(o-1)*2**(o-1)
                double curr_px_dst = PX_DST_MIN*(1 << k.octave);
                //k.sigma = MIN_SIGMA*std::pow(2, k.scale/NUM_SCL_PER_OCT)*std::pow(2,k.octave);
                k.sigma = (1 << k.octave) * MIN_SIGMA * std::pow(2, (k.scale + alfas[0])/NUM_SCL_PER_OCT);
                k.x = curr_px_dst * (k.m+alfas[1]);
                k.y = curr_px_dst * (k.n+alfas[2]);
                is_valid=true;
                break;
            }
        }
    return is_valid;
}

cv::Vec3d quadraticInterpolation(const Pyramid& DoG, Keypoint& k) {

    cv::Vec3d alfa_vals;
    const cv::Mat& img_prev = DoG.imgs[(k.octave*DoG.num_scales_per_oct)+k.scale-1];
    const cv::Mat& img_curr = DoG.imgs[(k.octave*DoG.num_scales_per_oct)+k.scale];
    const cv::Mat& img_next = DoG.imgs[(k.octave*DoG.num_scales_per_oct)+k.scale+1];

    double g1, g2, g3, h11, h12, h13, h22, h23, h33;

    g1 = 0.5*(img_next.at<double>(k.m, k.n) - img_prev.at<double>(k.m,k.n));
    g2 = 0.5*(img_curr.at<double>(k.m+1, k.n) - img_curr.at<double>(k.m-1,k.n));
    g3 = 0.5*(img_curr.at<double>(k.m, k.n+1) - img_curr.at<double>(k.m,k.n-1));

    h11 = img_next.at<double>(k.m, k.n) + img_prev.at<double>(k.m, k.n) - 2*img_curr.at<double>(k.m, k.n);
    h22 = img_curr.at<double>(k.m+1, k.n) + img_curr.at<double>(k.m-1, k.n) - 2*img_curr.at<double>(k.m, k.n);
    h33 = img_curr.at<double>(k.m, k.n+1) + img_curr.at<double>(k.m, k.n-1) - 2*img_curr.at<double>(k.m, k.n);
    h12 = 0.25*(img_next.at<double>(k.m+1, k.n) - img_next.at<double>(k.m-1, k.n) - img_prev.at<double>(k.m+1, k.n) + img_prev.at<double>(k.m-1, k.n));
    h13 = 0.25*(img_next.at<double>(k.m, k.n+1) - img_next.at<double>(k.m, k.n-1) - img_prev.at<double>(k.m, k.n+1) + img_prev.at<double>(k.m, k.n-1));
    h23 = 0.25*(img_curr.at<double>(k.m+1, k.n+1) - img_curr.at<double>(k.m+1, k.n-1) - img_curr.at<double>(k.m-1, k.n+1) + img_curr.at<double>(k.m-1, k.n-1));

    // Computation of the inverse of the Hessian matrix following the cofactor method
    double h11_inv, h12_inv, h13_inv, h22_inv, h23_inv, h33_inv;
    //double det_H = h11*(h22*h33 - h23*h23) - h12*(h12*h33 - h23*h13) + h13*(h12*h23 - h22*h13);
    double det_H = h11*h22*h33 - h11*h23*h23 - h12*h12*h33 + 2*h12*h23*h13 - h13*h22*h13;
    h11_inv = (h22*h33 - h23*h23) / det_H;
    h12_inv = (h13*h23 - h12*h33) / det_H;
    h13_inv = (h12*h23 - h13*h22) / det_H;
    h22_inv = (h11*h33 - h13*h13) / det_H;
    h23_inv = (h12*h13 - h11*h23) / det_H;
    h33_inv = (h11*h22 - h12*h12) / det_H;

    double alfa1, alfa2, alfa3;
    alfa1 = - h11_inv*g1 - h12_inv*g2 - h13_inv*g3;
    alfa2 = - h12_inv*g1 - h22_inv*g2 - h23_inv*g3;
    alfa3 = - h13_inv*g1 - h23_inv*g2 - h33_inv*g3;

    alfa_vals = {alfa1, alfa2, alfa3};

    k.omega = img_curr.at<double>(k.m, k.n) + 0.5*(alfa1*g1 + alfa2*g2 + alfa3*g3);

    return alfa_vals;
}

bool checkIfPointOnEdge(const Pyramid& DoG, const Keypoint& k, double C_edge) {

    double h11, h12, h22;
    const cv::Mat& img_curr = DoG.imgs[(k.octave*DoG.num_scales_per_oct)+k.scale];
    h11 = img_curr.at<double>(k.m+1, k.n) + img_curr.at<double>(k.m-1, k.n) - 2*img_curr.at<double>(k.m, k.n);
    h22 = img_curr.at<double>(k.m, k.n+1) + img_curr.at<double>(k.m, k.n-1) - 2*img_curr.at<double>(k.m, k.n);
    h12 = 0.25*(img_curr.at<double>(k.m+1, k.n+1) - img_curr.at<double>(k.m+1, k.n-1)
            - img_curr.at<double>(k.m-1, k.n+1) + img_curr.at<double>(k.m-1, k.n-1));

    double trace_H = h11+h22;
    double det_H = h11*h22 - h12*h12;
    double edgeness = (trace_H*trace_H)/det_H;
    double edgeness_threshold = ((C_edge+1)*(C_edge+1))/C_edge;

    if(edgeness < edgeness_threshold)
        return true;
    else
        return false;
}

std::vector<double> computeReferenceOrientation(Keypoint& k, const Pyramid& scaleSpaceGrads, double lamb_ori, double lamb_desc) {

        /* Writing the implementation as specified in the article "Anatomy of the SIFT method"
         * leads to memory inefficiency, i.e. unnecessarily wasted memory. Therefore, This implementation
         * will compute the reference orientation and descriptor for one keypoint at the time.*/
        std::vector<double> ref_oris{};
        double curr_pix_dst = PX_DST_MIN * std::pow(2, k.octave);
        double img_width = static_cast<double>(curr_pix_dst * scaleSpaceGrads.imgs[(k.octave * scaleSpaceGrads.num_scales_per_oct)+k.scale].rows);
        double img_height = static_cast<double>(curr_pix_dst * scaleSpaceGrads.imgs[(k.octave * scaleSpaceGrads.num_scales_per_oct)+k.scale].cols);
        double descr_patch_rad = std::sqrt(2) * lamb_desc * k.sigma;

        //Checking whether the keypoint is distant enough from the image borders
        //double min_dist = std::min({k.x, k.y, img_width - k.x, img_height - k.y});

        if (!(descr_patch_rad <= k.x && k.x <= img_width - descr_patch_rad &&
             descr_patch_rad <= k.y && k.y <= img_height - descr_patch_rad))
        //if(min_dist <= descr_patch_rad)
            return{};

        double gx, gy, grad_norm, exponent;
        int bin_num;
        double local_hist[N_BINS] = {};
        double ori_patch_rad = 3 * lamb_ori * k.sigma;
        int start_x = static_cast<int>(std::round((k.x - ori_patch_rad) / curr_pix_dst));
        int start_y = static_cast<int>(std::round((k.y - ori_patch_rad) / curr_pix_dst));
        int end_x = static_cast<int>(std::round((k.x + ori_patch_rad) / curr_pix_dst));
        int end_y = static_cast<int>(std::round((k.y + ori_patch_rad) / curr_pix_dst));

        for (int m = start_x; m <= end_x; ++m) {
            for (int n = start_y; n <= end_y; ++n) {
                // Whenever possible, the use of the power function should be avoided, as it's less efficient
                exponent = std::exp(-((m * curr_pix_dst - k.x) * (m * curr_pix_dst - k.x) +
                                            (n * curr_pix_dst - k.y) * (n * curr_pix_dst - k.y)) /
                                            (2 * (lamb_ori * k.sigma) * (lamb_ori * k.sigma)));
                // if possible, the use of the pow
                gx = scaleSpaceGrads.imgs[(k.octave*scaleSpaceGrads.num_scales_per_oct)+k.scale].at<cv::Vec2d>(m,n)[0];
                gy = scaleSpaceGrads.imgs[(k.octave*scaleSpaceGrads.num_scales_per_oct)+k.scale].at<cv::Vec2d>(m,n)[1];
                grad_norm = std::sqrt(gx*gx + gy*gy);

                bin_num = static_cast<int>(std::round((N_BINS/(2*M_PI) * std::fmod(std::atan2(gy, gx) +2*M_PI,2.0*M_PI))))%N_BINS;
                local_hist[bin_num] += exponent * grad_norm;
            }
        }

        // Smoothing the histogram using circular convolution
        double temp_hist[N_BINS] = {};
        for(int c = 0; c < 6; ++c) {
            for (int i = 0; i < N_BINS; ++i) {
                temp_hist[i] =
                        (local_hist[(i - 1 + N_BINS) % N_BINS] + local_hist[i] + local_hist[(i + 1) % N_BINS]) / 3.;
            }
            for (int i = 0; i < N_BINS; ++i) {
                local_hist[i] = temp_hist[i];
            }
        }

        // Extraction of reference orientation
        // First step: Find the maximum value in the histogram.
        double max_hist_val = 0;
        for(double h : local_hist) {
            if(h > max_hist_val) {
                max_hist_val = h;
            }
        }

        for(int i = 0; i < N_BINS; ++i) {
            // Still a minor degree of border wrapping
            if(local_hist[i] >= 0.8*max_hist_val) {
                double prev_element = local_hist[(i - 1 + N_BINS) % N_BINS];
                double next_element = local_hist[(i + 1) % N_BINS];
                if (local_hist[i] < prev_element || local_hist[i] < next_element)
                    continue;
                double ori_key = 2 * M_PI * (i + 1) / N_BINS + M_PI / N_BINS * (prev_element - next_element)/(prev_element - 2 * local_hist[i] +next_element);
                ref_oris.push_back(ori_key);

            }
        }
        return ref_oris;
}

std::vector<uint8_t> buildKeypointDescriptor(Keypoint& k, const double ori, const Pyramid& scaleSpaceGrads, double lamb_descr) {

    /* At this point the algorithm checks whether the keypoint is distant enough from
     * the image borders. This check has been already performed, in the previous step
     * of the algorithm. For more detail see algorithm 11 and 12 in the article "Anatomy
     * of the SIFT Method". */

    double k_dist = PX_DST_MIN * std::pow(2, k.octave);
    double rad = k.sigma * lamb_descr;

    cv::Mat grad_img = scaleSpaceGrads.imgs[(k.octave * scaleSpaceGrads.num_scales_per_oct) + k.scale];
    int start_x = static_cast<int>(std::round((k.x - (std::sqrt(2) * rad * (N_HISTS + 1.) / N_HISTS)) / k_dist));
    int end_x = static_cast<int>(std::round((k.x + (std::sqrt(2) * rad * (N_HISTS + 1.) / N_HISTS)) / k_dist));
    int start_y = static_cast<int>(std::round((k.y - (std::sqrt(2) * rad * (N_HISTS + 1.) / N_HISTS)) / k_dist));
    int end_y = static_cast<int>(std::round((k.y + (std::sqrt(2) * rad * (N_HISTS + 1.) / N_HISTS)) / k_dist));
    std::vector<uint8_t> descriptor{};

    //zero out the histogram
    double w_hist[N_HISTS*N_HISTS*N_ORI] = {};
    double sine_ori = std::sin(ori);
    double cosine_ori = std::cos(ori);
    for (int m = start_x; m <= end_x; ++m) {
        for (int n = start_y; n <= end_y; ++n) {
        // Compute
        double x_hat = ((m * k_dist - k.x) * cosine_ori + (n * k_dist - k.y) * sine_ori) / k.sigma;
        double y_hat = (-(m * k_dist - k.x) * sine_ori + (n * k_dist - k.y) * cosine_ori) / k.sigma;

        // Verify that the sample (m,n) is inside the normalized patch
        double max_dist = std::max(std::abs(x_hat), std::abs(y_hat));
        if (max_dist > lamb_descr * (N_HISTS + 1.) / N_HISTS)
            continue;

        // Compute normalized gradient orientation. We're adding 2*pi to ensure a positive result which
        // falls in the interval of [0, 2*pi], we could also add 4*pi.
        double exponent = -((m * k_dist - k.x) * (m * k_dist - k.x) + (n * k_dist - k.y) * (n * k_dist - k.y)) / (2 * (lamb_descr * k.sigma) * (lamb_descr * k.sigma));

        //extract the image gradients
        double gx = scaleSpaceGrads.imgs[(k.octave * scaleSpaceGrads.num_scales_per_oct) +
                                         k.scale].at<cv::Vec2d>(m, n)[0];
        double gy = scaleSpaceGrads.imgs[(k.octave * scaleSpaceGrads.num_scales_per_oct) +
                                         k.scale].at<cv::Vec2d>(m, n)[1];
        double norm_ori = std::fmod(atan2(gy, gx) - ori + 4 * M_PI,2 * M_PI);
        double grad_norm = std::sqrt(gx * gx + gy * gy);
        double contribution = std::exp(exponent) * grad_norm;

        // Updating the nearest histograms
        double x_hat_i, y_hat_j;
        for (int i = 1; i <= N_HISTS; ++i) {
            x_hat_i = (i - (1 + N_HISTS) / 2.) * 2 * lamb_descr / N_HISTS;
            if (std::abs(x_hat_i - x_hat) > 2 * lamb_descr / N_HISTS)
                continue;
            for (int j = 1; j <= N_HISTS; ++j) {
                y_hat_j = (j - (1 + N_HISTS) / 2.) * 2 * lamb_descr / N_HISTS;
                if (std::abs(y_hat_j - y_hat) > 2 * lamb_descr / N_HISTS)
                    continue;

                double xy_hat_hist = (1 - N_HISTS*0.5 / lamb_descr * std::abs(x_hat_i-x_hat)) *
                                     (1 - N_HISTS*0.5 / lamb_descr * std::abs(y_hat_j-y_hat));

                for (int k_ = 1; k_ <= N_ORI; ++k_) {
                    double ori_hat_k = 2 * M_PI * (k_ - 1.0) / N_ORI;
                    double ori_diff = std::fmod(ori_hat_k - norm_ori + 2 * M_PI, 2 * M_PI);

                    if (std::abs(ori_diff) >= (2 * M_PI) / N_ORI)
                        continue;

                    double ori_hist = 1. - N_ORI*0.5 / M_PI * std::abs(ori_diff);
                    w_hist[(i-1) * (N_HISTS * N_ORI) + (N_ORI * (j-1)) + (k_-1)] += xy_hat_hist * ori_hist * contribution;

                    }
                }
            }
        }
    }
    // Building the descriptor for the keypoint
    // the size of the descriptor
    int descr_size = N_HISTS * N_HISTS * N_ORI;

    //Computing the Euclidean norm of the vector
    double norm = 0.;
    for (int l = 0; l < descr_size; ++l) {
        norm += w_hist[l] * w_hist[l];
    }
    norm = std::sqrt(norm);

    double l2_norm = 0;
    for (int i = 0; i < descr_size; ++i) {
        w_hist[i] = std::min(w_hist[i], 0.2 * norm);
        l2_norm += w_hist[i] * w_hist[i];
    }
    l2_norm = std::sqrt(l2_norm);

    for (int j = 0; j < descr_size; ++j) {
        descriptor.push_back(
                std::min(static_cast<int>(std::floor((512 * w_hist[j]) / l2_norm)), 255));
    }
    return descriptor;
}

// The main function fusing all previous algorithms into one pipeline
keypoints detect_keypoints(cv::Mat img, double lamd_descr, double lamb_ori) {

    //cv::resize(img,img, cv::Size(700,700), 0,0,cv::INTER_CUBIC);
    cv::Mat gray_image;
    cv::cvtColor(img, gray_image, cv::COLOR_BGR2GRAY);

    gray_image.convertTo(gray_image, CV_64F);
    cv::normalize(gray_image, gray_image, 0, 1, cv::NORM_MINMAX, CV_64F);
    gray_image = gray_image.t();
    Pyramid pyramid = computeGaussianPyramid(gray_image);

    Pyramid DoG = computeDoGPyramid(pyramid);
    keypoints kp_points = locateExtrema(DoG);
    cv::Mat img_clone = img.clone();
    drawKeypoints(img_clone, kp_points);
    Pyramid gradPyr = computeGradientImages(pyramid);

    keypoints kpoints{};
    std::ofstream outFile("../ori_values.txt");
    //auto *weighted_historgrams = static_cast<double *>(malloc(N_HISTS * N_HISTS * N_ORI * sizeof(double)));
    for(Keypoint& kp : kp_points) {

        // Gathering orientations for each keypoint
        std::vector<double> oris = computeReferenceOrientation(kp, gradPyr, LAMB_ORI, LAMB_DESC);
        /* If a keypoint contains more than one reference orientation,
         * multiple keypoints will be created at that exact locations,
         * all with their own reference. For reference see the original
         * SIFT paper page 13. */
        for (double ori: oris) {
            std::vector<uint8_t> descriptor = buildKeypointDescriptor(kp, ori, gradPyr, LAMB_DESC);
            kp.descriptor = descriptor;
            kpoints.push_back(kp);
            outFile << ori << "\n";
        }
    }
    outFile.close();
    return kpoints;
}

// Matching of keypoints following the procedure in article "Anatomy of the SIFT method"
matches match_keypoints(const keypoints& keypoints_img1, const keypoints& keypoints_img2, double rThr, int dThr) {
    matches key_matches{};
    for(const auto& k1 : keypoints_img1) {
        double min_dist1 = 1000000.;
        double min_dist2 = 1000000.;
        Keypoint min_k2{};
        for(const auto& k2 : keypoints_img2) {
            double dist = computeEuclidenDist(k1, k2);
            if(dist < min_dist1){
                min_dist2 = min_dist1;
                min_dist1 = dist;
                min_k2 = k2;
            } else if (min_dist1 <= dist && dist < min_dist2) {
                min_dist2 = dist;
            }
        }
        if(min_dist1 < rThr * min_dist2 && min_dist1 < dThr) {
            std::pair<Keypoint, Keypoint> match = {k1, min_k2};
            key_matches.push_back(match);
        }
    }
    return key_matches;
}

double computeEuclidenDist(const Keypoint& k1, const Keypoint& k2) {

    double total_dist = 0.;
    for(int d = 0; d < 128; ++d) {
        int a =  (k1.descriptor[d] - k2.descriptor[d]);
        total_dist += a*a;
    }
    return std::sqrt(total_dist);
}

void drawMatchesKey(const cv::Mat& img1, const cv::Mat& img2, matches& key_matches) {
    // Resize images to around 600x600
    cv::Size newSize(700, 700);
    //cv::Mat resizedImg1, resizedImg2;
    //cv::resize(img1, resizedImg1, newSize);
    //cv::resize(img2, resizedImg2, newSize);

    // Concatenate the two resized images horizontally
    int maxHeight = std::max(img1.rows, img2.rows);

    // Create a blank image with the maximum height and the sum of widths of the two images
    cv::Mat concatImage(maxHeight, img1.cols + img2.cols, CV_8UC3, cv::Scalar(0, 0, 0));

    // Copy the first image to the left side of the blank image
    img1.copyTo(concatImage(cv::Rect(0, 0, img1.cols, img1.rows)));

    // Copy the second image to the right side of the blank image
    img2.copyTo(concatImage(cv::Rect(img1.cols, 0, img2.cols, img2.rows)));

    // Draw circles at matching keypoints
    for (auto& match : key_matches) {
        Keypoint& k1 = match.first;
        Keypoint& k2 = match.second;

        // Draw circle around the keypoints
        cv::circle(concatImage, cv::Point(k1.x, k1.y), 5, cv::Scalar(0, 255, 0), 2);
        cv::circle(concatImage, cv::Point(k2.x+img1.cols, k2.y), 5, cv::Scalar(0, 255, 0), 2);
    }

    // Display the concatenated image with matches
    cv::imshow("Matches", concatImage);
    cv::waitKey(0);
}

void drawLine(cv::Mat& img, int x1, int y1, int x2, int y2) {
    // Ensure x1 <= x2 for consistency
    if (x2 < x1) {
        std::swap(x1, x2);
        std::swap(y1, y2);
    }

    int dx = x2 - x1;
    int dy = y2 - y1;

    for (int x = x1; x <= x2; x++) {
        int y = y1 + dy * (x - x1) / dx;

        if (img.channels() == 3) {
            // Draw green line on color image
            img.at<cv::Vec3b>(y, x) = cv::Vec3b(0, 255, 0);
        } else {
            // Draw white line on grayscale image
            img.at<uchar>(y, x) = 255;
        }
    }
}