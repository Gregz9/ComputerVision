#include "sift.h"
#include "filters.h"

Pyramid computeGaussianPyramid(cv::Mat img){
    cv::resize(img, img, cv::Size(img.cols*2, img.rows*2),0,0, cv::INTER_LINEAR);
    // Computing the standard deviation of the gaussian kernel
    // For the very first image, init_sigma is equivalent to:

    double std_dev = std::sqrt((MIN_SIGMA*MIN_SIGMA) - (INP_SIGMA*INP_SIGMA));
    std_dev /= PX_DST_MIN;

    cv::Mat kernel = create1DGaussKernel(std_dev);
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

        for (int j = 1; j < std_devs.size(); j++) {
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
            std::vector<cv::Mat>(scale_space.imgs.size())
    };

    for(int o = 0; o < scale_space.num_oct; ++o) {
        for(int s = 0; s < scale_space.num_scales_per_oct; ++s) {
            cv::Mat curr_img = scale_space.imgs[(o*scale_space.num_scales_per_oct)+s];
            cv::Mat grad_img(curr_img.size(), CV_64FC2);
            for(int i = 1; i < curr_img.rows-1; ++i) {
                for(int j = 1; j < curr_img.cols-1; ++j) {
                    double gx = 0.5*(curr_img.at<double>(i+1, j) - curr_img.at<double>(i-1, j));
                    double gy = 0.5*(curr_img.at<double>(i, j+1) - curr_img.at<double>(i, j-1));
                    cv::Mat grads{gx, gy};
                    grad_img.at<cv::Vec2d>(i,j) = grads;
                }
            }
            ScaleSpaceGradients.imgs.push_back(grad_img);
        }
    }
    return ScaleSpaceGradients;
}

keypoints locateExtrema(const Pyramid dog, double C_dog, double C_edge) {
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
                        KeyPoint k = {i, j, o, s};
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

bool keypointRefinement(const Pyramid& DoG, KeyPoint& k) {
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
                k.sigma = curr_px_dst * MIN_SIGMA * std::pow(2, (k.scale)/NUM_SCL_PER_OCT);
                k.x = curr_px_dst * (k.n);
                k.y = curr_px_dst * (k.m);
                is_valid=true;
                break;
            }
        }
    return is_valid;
}

cv::Vec3d quadraticInterpolation(const Pyramid& DoG, KeyPoint& k) {

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

bool checkIfPointOnEdge(Pyramid DoG, KeyPoint k, double C_edge) {

    double h11, h12, h22;
    const cv::Mat& img_curr = DoG.imgs[(k.octave*DoG.num_scales_per_oct)+k.scale];
    h11 = img_curr.at<double>(k.m+1, k.n) + img_curr.at<double>(k.m-1, k.n) - 2*img_curr.at<double>(k.m, k.n);
    h22 = img_curr.at<double>(k.m, k.n+1) + img_curr.at<double>(k.m, k.n-1) - 2*img_curr.at<double>(k.m, k.n);
    h12 = 0.25f*(img_curr.at<double>(k.m+1, k.n+1) - img_curr.at<double>(k.m+1, k.n-1)
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

void computeReferenceOrientation(KeyPoint& k, const Pyramid& scaleSpaceGrads, double lamb_ori, double lamb_desc) {

        /* Writing the implementation as specified in the article "Anatomy of the SIFT method"
         * leads to memory inefficiency, i.e. unnecessarily wasted memory. Therefore, This implementation
         * will compute the reference orientation and descriptor for one keypoint at the time.*/

        double curr_pix_dst = PX_DST_MIN * pow(2, k.octave);
        double img_width = curr_pix_dst * scaleSpaceGrads.imgs[(k.octave * scaleSpaceGrads.num_scales_per_oct)].cols;
        double img_height = curr_pix_dst * scaleSpaceGrads.imgs[(k.octave * scaleSpaceGrads.num_scales_per_oct)].rows;
        double descr_patch_rad = std::sqrt(2) * lamb_desc * k.sigma;

        //Checking whether the keypoint is distant enough from the image borders
        if (!(descr_patch_rad <= k.x && k.x <= img_width - descr_patch_rad &&
              descr_patch_rad <= k.y && k.y <= img_height - descr_patch_rad)) {
                return;
        }
        double gx, gy, grad_norm, grad_ori, exponent;
        int bin_num;
        double local_hist[N_BINS];
        double ori_patch_rad = 3 * lamb_ori * k.sigma;
        int start_x = static_cast<int>(round((k.x - ori_patch_rad) / curr_pix_dst));
        int start_y = static_cast<int>(round((k.y - ori_patch_rad) / curr_pix_dst));
        int end_x = static_cast<int>(round((k.x + ori_patch_rad) / curr_pix_dst));
        int end_y = static_cast<int>(round((k.y + ori_patch_rad) / curr_pix_dst));

        for (int m = start_x; m <= end_x; ++m) {
            for (int n = start_y; n <= end_y; ++n) {
                // Whenever possible, the use of the power function should be avoided, as it's less efficient
                exponent = std::exp(-((m * curr_pix_dst - k.x) * (m * curr_pix_dst - k.x) +
                                            (n * curr_pix_dst - k.y) * (n * curr_pix_dst - k.y)) /
                                           (2 * (lamb_ori * k.sigma) * (lamb_ori * k.sigma)));
                // if possible, the use of the pow
                gx = scaleSpaceGrads.imgs[(k.octave*scaleSpaceGrads.num_scales_per_oct)+k.scale].at<cv::Vec2d>(m,n)[0];
                gy = scaleSpaceGrads.imgs[(k.octave*scaleSpaceGrads.num_scales_per_oct)+k.scale].at<cv::Vec2d>(m,n)[1];
                grad_norm = std::sqrt((gx*gx) + (gy*gy));

                bin_num = static_cast<int>(round((N_BINS/(2*M_PI)) * atan2(gy, gx) * std::fmod((2*M_PI),2.0)));
                local_hist[bin_num] = exponent * grad_norm;
            }
        }
        // Smoothing the histogram using circular convolution
        double temp_hist[N_BINS];
        for(int c = 0; c < 6; ++c) {
            for (int i = 0; i < N_BINS; ++i) {
                temp_hist[i] =
                        (local_hist[((i - 1) + N_BINS) % N_BINS] + local_hist[i] + local_hist[(i + 1) % N_BINS]) / 3.;
            }
            for (int i = 0; i < N_BINS; ++i) {
                local_hist[i] = temp_hist[i];
            }
        }

        // Extraction of reference orientation
        // First step: Find the maximum value in the histogram.
        double max_hist_val{0};
        for(double h : local_hist) {
            if(h > max_hist_val)
                max_hist_val = h;
        }

        std::vector<double> orientations;
        for(int i = 0; i < N_BINS; ++i) {
            // Still a minor degree of border wrapping
            double prev_element = local_hist[((i-1)+N_BINS)%N_BINS];
            double next_element = local_hist[(i+1)%N_BINS];
            if(local_hist[i] > prev_element &&
                local_hist[i] > next_element &&
                local_hist[i] < 0.8*max_hist_val) {
                double ori_key = (2*M_PI*(i-1))/N_BINS + (M_PI/N_BINS) * ((prev_element - next_element)/(prev_element - 2*local_hist[i] + next_element));
                k.ref_oris.push_back(ori_key);
            }
        }
}

void buildKeypointDescriptor(KeyPoint& k, const Pyramid& scaleSpaceGrads, double lamb_descr) {

    // Required values
    double k_dist = PX_DST_MIN * std::pow(2, k.octave);
    double rad = k.sigma * lamb_descr;

    /* At this point the algorithm checks whether the keypoint is distant enough from
     * the image borders. This check has been already performed, in the previous step
     * of the algorithm. For more detail see algorithm 11 and 12 in the article "Anatomy
     * of the SIFT Method". */

    // Initializing the histograms
    double* weighted_historgrams = static_cast<double *>(malloc(
            N_HISTS * N_HISTS * N_ORI * sizeof weighted_historgrams));

    cv::Mat grad_img = scaleSpaceGrads.imgs[(k.octave*scaleSpaceGrads.num_scales_per_oct)+k.scale];
    int start_x = static_cast<int>(std::round((k.x - std::sqrt(2)*rad*((N_HISTS+1.)/N_HISTS))/k_dist));
    int end_x = static_cast<int>(std::round((k.x + std::sqrt(2)*rad*((N_HISTS+1.)/N_HISTS))/k_dist));
    int start_y = static_cast<int>(std::round((k.y - std::sqrt(2)*rad*((N_HISTS+1.)/N_HISTS))/k_dist));
    int end_y = static_cast<int>(std::round((k.y + std::sqrt(2)*rad*((N_HISTS+1.)/N_HISTS))/k_dist));

    for(double ori : k.ref_oris) {
        double sine_ori = std::sin(ori);
        double cosine_ori = std::cos(ori);
        for(int m = start_x; m <= end_x; ++m) {
            for(int n = start_y; n <= end_y; ++n) {
                // Compute
                double x_hat = (((m*k_dist - k.x)*cosine_ori) + ((n*k_dist - k.y)*sine_ori))/k.sigma;
                double y_hat = (((m*k_dist - k.x)*cosine_ori) + ((n*k_dist - k.y)*sine_ori))/k.sigma;

                // Verify that the sample (m,n) is inside the normalized patch
                double max_dist = std::max(std::abs(x_hat), std::abs(y_hat));
                if(max_dist > lamb_descr * ((N_HISTS+1.)/N_HISTS))
                    continue;

                // Compute normalized gradient orientation. We're adding 2*pi to ensure a positive result which
                // falls in the interval of [0, 2*pi], we could also add 4*pi.
                double norm_ori = std::fmod(atan2(grad_img.at<cv::Vec2d>(m,n)[1], grad_img.at<cv::Vec2d>(m,n)[1]) - ori + 2*M_PI, 2*M_PI);
                double exponent = (((m*k_dist - k.x)*(m*k_dist - k.x)) + ((n*k_dist - k.y)*(n*k_dist - k.y)))/(2*(lamb_descr*k.sigma)*(lamb_descr*k.sigma));

                //extract the image gradients
                double gx = scaleSpaceGrads.imgs[(k.octave*scaleSpaceGrads.num_scales_per_oct)+k.scale].at<cv::Vec2d>(m,n)[0];
                double gy = scaleSpaceGrads.imgs[(k.octave*scaleSpaceGrads.num_scales_per_oct)+k.scale].at<cv::Vec2d>(m,n)[1];
                double grad_norm = std::sqrt((gx*gx) + (gy*gy));
                double contribution = std::exp(-exponent)*grad_norm;

                // Updating the nearest histograms
                double x_hat_i, y_hat_j;
                for(int i = 0; i < N_HISTS; ++i) {
                    x_hat_i = (i - ((1 + N_HISTS)/2))*((2*lamb_descr)/N_HISTS);
                    if (std::abs(x_hat_i - x_hat) > ((2*lamb_descr)/N_HISTS))
                        continue;
                    for(int j = 0; j < N_HISTS; ++j) {
                        y_hat_j = (j - ((1 + N_HISTS)/2))*((2*lamb_descr)/N_HISTS);
                        if(std::abs(y_hat_j - y_hat) > ((2*lamb_descr)/N_HISTS))
                            continue;

                        double xy_hat_hist = (1 - (N_HISTS/(2*lamb_descr))*std::abs(x_hat - x_hat_i)) * (1 - (N_HISTS/(2*lamb_descr))*std::abs(y_hat - y_hat_j));
                        for(int k_ = 0; k_ < N_ORI; ++k_) {
                            double ori_hat_k = (2*M_PI*(k_-1.0))/N_ORI;
                            double ori_hist = (1-(N_ORI/(2*M_PI)))*std::abs(ori_hat_k);
                            if(std::fmod(std::abs(norm_ori - ori_hat_k+2*M_PI), 2*M_PI) >= (2*M_PI)/N_ORI)
                                continue;

                            weighted_historgrams[i*(N_HISTS*N_ORI)+(N_ORI*j)+k_] += xy_hat_hist*ori_hist*contribution;

                        }
                    }
                }
            }
        }
    }
    // Building the descriptor for the keypoint
    // the size of the descriptor
    int descr_size = N_HISTS*N_HISTS*N_ORI;

    //Computing the Euclidean norm of the vector
    double norm = 0.;
    for(int l = 0; l < descr_size; ++l) {
        norm += weighted_historgrams[l]*weighted_historgrams[l];
    }
    norm = std::sqrt(norm);

    double l2_norm = 0;
    for(int i = 0; i < descr_size; ++i) {
        weighted_historgrams[i] = std::min(weighted_historgrams[i], 0.2*norm);
        l2_norm += weighted_historgrams[i]*weighted_historgrams[i];
    }
    l2_norm = std::sqrt(l2_norm);

    for(int j = 0; j < descr_size; ++j) {
        k.descriptor.push_back(std::min(static_cast<int>(std::floor((512*weighted_historgrams[j])/l2_norm)), 255));
    }

    free(weighted_historgrams);
}