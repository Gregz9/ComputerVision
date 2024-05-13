#include "sift.h"
#include "filters.h"

Pyramid computeGaussianPyramid(cv::Mat img){
    cv::resize(img, img, cv::Size(img.rows*2, img.cols*2), cv::INTER_LINEAR);
    // Computing the standard deviation of the gaussian kernel
    // For the very first image, init_sigma is equivalent to:
    float std_dev = std::sqrt((MIN_SIGMA*MIN_SIGMA) - (INP_SIGMA*INP_SIGMA));
    std_dev /= PX_DST_MIN;

    cv::Mat kernel = create1DGaussKernel(std_dev);
    cv::sepFilter2D(img, img, CV_32F, kernel, kernel);

    int num_scales = NUM_SCL_PER_OCT + 3;
    float min_coef = MIN_SIGMA / PX_DST_MIN;
    auto k = static_cast<float>(std::pow(2, 1.0/NUM_SCL_PER_OCT));

    std::vector<float> std_devs {min_coef};
    Pyramid pyramid;
    pyramid.num_scales_per_oct = num_scales;

    // Computing standard deviations
    for (int i=1; i<num_scales; ++i){
        auto prev_sigma = static_cast<float>(std::pow(k, i-1) * min_coef);
        float sigma = k * prev_sigma;
        std_devs.push_back(std::sqrt((sigma*sigma) - (prev_sigma*prev_sigma)));
    }

    pyramid.imgs.reserve(num_scales*NUM_OCT);
    for (int i=0; i<NUM_OCT; i++){
        pyramid.imgs.push_back(img.clone());

        for(int j=1; j < std_devs.size(); j++){
            const cv::Mat scale_img = pyramid.imgs.at(i*num_scales).clone();
            cv::Mat blur_kernel = create1DGaussKernel(std_devs[j]);
            cv::sepFilter2D(scale_img, scale_img, CV_32F, blur_kernel, blur_kernel);

            pyramid.imgs.push_back(scale_img.clone());
        }
        const cv::Mat next_img = pyramid.imgs.at((i*num_scales)+NUM_SCL_PER_OCT);
        cv::resize(next_img, img, cv::Size(img.rows/2, img.cols/2), cv::INTER_LINEAR);

    }
    return pyramid;
}

Pyramid computeDoGPyramid(const Pyramid gauss) {

    Pyramid dog;
    dog.num_scales_per_oct = gauss.num_scales_per_oct - 1;
    dog.imgs.reserve(dog.num_scales_per_oct*dog.num_oct);

    for (int o = 0; o < dog.num_oct; ++o) {
        for (int s = 1; s < gauss.num_scales_per_oct; ++s) { // Start from s = 0
            cv::Mat diff_img = gauss.imgs[(o * gauss.num_scales_per_oct) + s] - gauss.imgs[(o * gauss.num_scales_per_oct) + s-1];
            dog.imgs.push_back(diff_img.clone());
        }
    }
    return dog;
}

keypoints locateExtrema(const Pyramid dog, float C_dog, float C_edge) {
    // Impossible to know how many keypoint candidates there'll
    // be, hence cannot reserve a specific size for the vector.
    keypoints k_points{};
    for(int o = 0; o < dog.num_oct; ++o) {
        // DoG pyramid has already num_scales-1, hence only 1 subtracted
        for(int s=1; s < dog.num_scales_per_oct-1; ++s) {

            const cv::Mat& img_DoG = dog.imgs[(o*dog.num_scales_per_oct)+s];
            const cv::Mat& prev_img_DoG = dog.imgs[(o*dog.num_scales_per_oct)+s-1];
            const cv::Mat& next_img_DoG = dog.imgs[(o*dog.num_scales_per_oct)+s+1];

            for(int i = 1; i < img_DoG.rows -1; ++i) {
                for(int j =1; j < img_DoG.cols -1; ++j) {
                    float current_pixel = img_DoG.at<float>(i,j);
                    // It's more efficient to check for the threshold criteria here, rather than later.
                    if(std::abs(current_pixel) < 0.8f*C_dog) {
                        continue;
                    }

                    bool max = true, min = true;

                    for(int x = -1; x<2; ++x) {
                        for (int y = -1; y < 2; ++y) {
                            float neighbor_pixel_curr = img_DoG.at<float>(i + x, j + y);
                            float neighbor_pixel_prev = prev_img_DoG.at<float>(i + x, j + y);
                            float neighbor_pixel_next = next_img_DoG.at<float>(i + x, j + y);

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
                        if(valid)
                            k_points.push_back(k);
                    }

                    // Refinement could be done here, i.e. keypoint interpolation
                }
            }
        }
    }
    return k_points;
}

bool keypointRefinement(const Pyramid DoG, KeyPoint k) {
        bool is_valid = false;
        int t = 0;
        while(t++ < MAX_INTER_ITERS) {
            // Quadratic interpolation
            cv::Vec3f alfas = quadraticInterpolation(DoG, k);
            k.scale += static_cast<int>(round(alfas[0]));
            k.m += static_cast<int>(round(alfas[1]));
            k.n += static_cast<int>(round(alfas[2]));
            if(k.scale >= DoG.num_oct-1 || k.scale < 1)
                break;

            float quad_extremum = std::max({abs(alfas[0]), abs(alfas[1]), abs(alfas[2])});
            bool not_edge = checkIfPointOnEdge(DoG, k, EDGE_THR);
            bool valid_contr = abs(k.omega) > DOG_THR;

            if (quad_extremum < 0.6 && valid_contr && not_edge) {

                // Continuous values of sigma and image coordinates. RMB! PX_DIST(o) = PX_DST(o-1)*2**(o-1)
                float curr_px_dist = PX_DST_MIN * static_cast<float>(pow(2, k.octave));
                k.sigma = MIN_SIGMA*static_cast<float>(pow(2, k.scale/NUM_SCL_PER_OCT));
                k.x = curr_px_dist * k.m;
                k.y = curr_px_dist * k.n;
                is_valid=true;
                break;
            }
        }

    return is_valid;
}

cv::Vec3f quadraticInterpolation(Pyramid DoG, KeyPoint& k) {

    cv::Vec3f alfa_vals;
    cv::Mat img_prev = DoG.imgs[(k.octave*DoG.num_scales_per_oct)+k.scale-1];
    cv::Mat img_curr = DoG.imgs[(k.octave*DoG.num_scales_per_oct)+k.scale];
    cv::Mat img_next = DoG.imgs[(k.octave*DoG.num_scales_per_oct)+k.scale+1];

    float g1, g2, g3, h11, h12, h13, h22, h23, h33;

    g1 = 0.5f*(img_next.at<float>(k.m, k.n) - img_prev.at<float>(k.m,k.n));
    g2 = 0.5f*(img_curr.at<float>(k.m+1, k.n) - img_curr.at<float>(k.m-1,k.n));
    g3 = 0.5f*(img_curr.at<float>(k.m, k.n+1) - img_curr.at<float>(k.m,k.n-1));

    h11 = img_next.at<float>(k.m, k.n) + img_prev.at<float>(k.m, k.n) - 2*img_curr.at<float>(k.m, k.n);
    h22 = img_curr.at<float>(k.m+1, k.n) + img_curr.at<float>(k.m-1, k.n) - 2*img_curr.at<float>(k.m, k.n);
    h33 = img_curr.at<float>(k.m, k.n+1) + img_curr.at<float>(k.m, k.n-1) - 2*img_curr.at<float>(k.m, k.n);
    h12 = 0.25f*(img_next.at<float>(k.m+1, k.n) - img_next.at<float>(k.m-1, k.n) - img_prev.at<float>(k.m+1, k.n) + img_prev.at<float>(k.m-1, k.n));
    h13 = 0.25f*(img_next.at<float>(k.m, k.n+1) - img_next.at<float>(k.m, k.n-1) - img_prev.at<float>(k.m, k.n+1) + img_prev.at<float>(k.m, k.n-1));
    h23 = 0.25f*(img_curr.at<float>(k.m+1, k.n+1) - img_curr.at<float>(k.m+1, k.n-1) - img_curr.at<float>(k.m-1, k.n+1) + img_curr.at<float>(k.m-1, k.n-1));

    // Computation of the inverse of the Hessian matrix following the cofactor method
    float h11_inv, h12_inv, h13_inv, h22_inv, h23_inv, h33_inv;
    //float det_H = h11*(h22*h33 - h23*h23) - h12*(h12*h33 - h23*h13) + h13*(h12*h23 - h22*h13);
    float det_H = h11*h22*h33 - h11*h23*h23 - h12*h12*h33 + 2*h12*h23*h13 - h13*h22*h13;
    h11_inv = (h22*h33 - h23*h23) / det_H;
    h12_inv = (h23*h13 - h12*h33) / det_H;
    h13_inv = (h12*h23 - h22*h13) / det_H;
    h22_inv = (h11*h33 - h13*h13) / det_H;
    h23_inv = (h12*h13 - h11*h23) / det_H;
    h33_inv = (h11*h22 - h12*h12) / det_H;

    float alfa1, alfa2, alfa3;
    alfa1 = - h11_inv*g1 - h12_inv*g2 - h13_inv*g3;
    alfa2 = - h12_inv*g1 - h22_inv*g2 - h23_inv*g3;
    alfa3 = - h13_inv*g1 - h23_inv*g2 - h33_inv*g3;

    alfa_vals = {alfa1, alfa2, alfa3};

    k.omega = img_curr.at<float>(k.m, k.n) + 0.5f*(alfa1*g1 + alfa2*g2 + alfa3*g3);

    return alfa_vals;
}

bool checkIfPointOnEdge(Pyramid DoG, KeyPoint k, float C_edge) {

    float h11, h12, h22;
    cv::Mat img_curr = DoG.imgs[(k.octave*DoG.num_scales_per_oct)+k.scale];
    h11 = img_curr.at<float>(k.m+1, k.n) + img_curr.at<float>(k.m-1, k.n) - 2*img_curr.at<float>(k.m, k.n);
    h22 = img_curr.at<float>(k.m, k.n+1) + img_curr.at<float>(k.m, k.n-1) - 2*img_curr.at<float>(k.m, k.n);
    h12 = 0.25f*(img_curr.at<float>(k.m+1, k.n+1) + img_curr.at<float>(k.m+1, k.n-1) - img_curr.at<float>(k.m-1, k.n+1) + img_curr.at<float>(k.m-1, k.n-1));

    float trace_H = h11+h22;
    float det_H = h11*h22 - h12*h12;
    float edgeness = (trace_H*trace_H)/det_H;
    float edgeness_threshold = ((C_edge+1)*(C_edge+1))/C_edge;

    if(edgeness < edgeness_threshold)
        return true;
    else
        return false;
}