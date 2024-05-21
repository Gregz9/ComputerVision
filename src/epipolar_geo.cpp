#include "epipolar_geo.h"
#include "sift.h"

//Utility function for constructing the A matrix for least squares
void vector2mat(const std::vector<cv::Vec2f>& vec, cv::Mat_<float>& mat) {
    mat = cv::Mat_<float>(2, static_cast<int>(vec.size()));
    for(int i = 0; i < static_cast<int>(vec.size()); ++i){
        mat(0, i) = vec[i][0];
        mat(1, i) = vec[i][1];
    }
}

cv::Mat estimateFundamentalMatrix8Pts(const std::vector<cv::Vec2f>& x1, const std::vector<cv::Vec2f>& x2) {
    std::vector<cv::Vec2f> norm_x1, norm_x2;
    cv::Matx33f T1, T2;

    T1 = standarizeCoords(x1);
    T2 = standarizeCoords(x2);

    cv::Mat_<float> matx1, matx2;
    vector2mat(x1, matx1);
    vector2mat

    return {};
}

cv::Matx33f standarizeCoords(const std::vector<cv::Vec2f>& points){

    std::vector<cv::Vec2d> norm_points(points.size());
    cv::Vec2d p1 = {0, 0};
    // Computing the average over coordinates
    for(auto& p : points) {
        p1[0] += p[0];
        p1[1] += p[1];
    }
    p1 /= static_cast<int>(points.size());

    /* Translating the points and computing the average distance
     * from the origin */
    double d1 = 0.;
    for(int i = 0; i < static_cast<int>(points.size()); ++i) {
        norm_points[i][0] = points[i][0] - p1[0];
        norm_points[i][1] = points[i][0] - p1[1];
        d1 += std::sqrt(norm_points[i][0]*norm_points[i][0] + norm_points[i][1]*norm_points[i][1]);
    }
    double avgDistance = d1 / static_cast<int>(points.size());
    double scale = std::sqrt(2)/avgDistance;

    for(int i = 0; i < static_cast<int>(points.size()); ++i) {
        norm_points[i] *= scale;
    }

    // Orginization of the normalization coefficient in a 3D
    // Transformation matrix
    cv::Matx33f T;
    T << scale, 0, -scale * p1[0],
        0,      0, -scale * p1[1],
        0,      0,  1;

    return T;

}