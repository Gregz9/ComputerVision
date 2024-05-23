#include "epipolar_geo.h"


//Utility function for constructing the A matrix for least squares
void vector2mat(const std::vector<Eigen::Vector2f>& vec, Eigen::MatrixXf& mat) {
    for(int i = 0; i < static_cast<int>(vec.size()); ++i){
        mat(0, i) = vec[i][0];
        mat(1, i) = vec[i][1];
    }
}

Eigen::Matrix3f estimateFundamentalMatrix8Pts(const std::vector<cv::Vec2f>& x1, const std::vector<cv::Vec2f>& x2) {

    Eigen::Matrix3f F;
    std::vector<Eigen::Vector2f> p1(x1.size()), p2(x2.size());
    for(int i = 0; i < 8; ++i) {
        p1[i][0] = x1[i][0];
        p1[i][1] = x1[i][1];
        p2[i][0] = x1[i][0];
        p2[i][1] = x1[i][1];
    }
    std::vector<Eigen::Vector2f> norm_x1, norm_x2;
    Eigen::Matrix3f T1, T2;

    T1 = standarizeCoords(p1);
    T2 = standarizeCoords(p2);

    Eigen::MatrixXf matx1, matx2;
    vector2mat(p1, matx1);
    vector2mat(p2, matx2);

    Eigen::MatrixXf ones(1, 8);
    ones.setOnes();

    Eigen::MatrixXf A(9, 8);
    A << matx2.row(0) * matx1.row(0),
        matx2.row(0) *  matx1.row(1),
        matx2.row(0),
        matx2.row(1) * matx1.row(0),
        matx2.row(1) * matx1.row(1),
        matx2.row(1),
        matx1.row(0),
        matx1.row(1),
        ones;
    A = A.transpose().eval();
    Eigen::VectorXf minSingVVec(9);
    /* Estimating the Fundamental matrix using the right singular vector corresponding to
    * the smallest eigenvalue */
    Eigen::JacobiSVD<Eigen::Matrix<float, Eigen::Dynamic, 9>> svdA(A, Eigen::ComputeFullV);
    minSingVVec = svdA.matrixV().col(8);

    F << minSingVVec(0), minSingVVec(1), minSingVVec(2),
         minSingVVec(3), minSingVVec(4), minSingVVec(5),
         minSingVVec(6), minSingVVec(7), minSingVVec(8);

    //Performing SVD once again on the initial estimate of the fundamental matrix
    Eigen::JacobiSVD<Eigen::Matrix3f> svdF(F, Eigen::ComputeFullU | Eigen::ComputeFullV);
    //Enforcing a zero determinant by setting hte smallest eigenvalue to zero
    Eigen::Vector3f sing_vals = svdF.singularValues();
    sing_vals(2) = 0.0;
    F = svdF.matrixU() * sing_vals.asDiagonal() * svdF.matrixV();
    return T2.transpose()*F*T1;
}

Eigen::Matrix3f estimateFundamentalMatrix7Pts(const std::vector<cv::Vec2f>& x1, const std::vector<cv::Vec2f>& x2) {

    Eigen::Matrix3f F;
    std::vector<Eigen::Vector2f> p1(x1.size()), p2(x2.size());
    for(int i = 0; i < static_cast<int>(x1.size()); ++i) {
        p1[i][0] = x1[i][0];
        p1[i][1] = x1[i][1];
        p2[i][0] = x1[i][0];
        p2[i][1] = x1[i][1];
    }

    Eigen::MatrixXf matx1, matx2;
    vector2mat(p1, matx1);
    vector2mat(p2, matx2);
    Eigen::MatrixXf ones = Eigen::MatrixXf(1,7);

    Eigen::MatrixXf A(9, 7);
    Eigen::MatrixXf A(9, 8);
    A << matx2.row(0) * matx1.row(0),
            matx2.row(0) *  matx1.row(1),
            matx2.row(0),
            matx2.row(1) * matx1.row(0),
            matx2.row(1) * matx1.row(1),
            matx2.row(1),
            matx1.row(0),
            matx1.row(1),
            ones;
    A = A.transpose().eval();
    Eigen::JacobiSVD<Eigen::Matrix<float, Eigen::Dynamic, 9>> svdA(A, Eigen::ComputeFullV);
    Eigen::VectorXf secMinSingVVec(8);
    Eigen::VectorXf minSingVVec(9);
}

Eigen::Matrix3f standarizeCoords(const std::vector<Eigen::Vector2f>& points){

    std::vector<Eigen::Vector2f> norm_points(points.size());
    Eigen::Vector2f p1 = {0, 0};
    // Computing the average over coordinates
    for(auto& p : points) {
        p1[0] += p[0];
        p1[1] += p[1];
    }
    p1 /= static_cast<float>(points.size());

    /* Translating the points and computing the average distance
     * from the origin */
    float d1 = 0.;
    for(int i = 0; i < static_cast<int>(points.size()); ++i) {
        norm_points[i][0] = points[i][0] - p1[0];
        norm_points[i][1] = points[i][0] - p1[1];
        d1 += std::sqrt(norm_points[i][0]*norm_points[i][0] + norm_points[i][1]*norm_points[i][1]);
    }
    float avgDistance = d1 / static_cast<float>(points.size());
    auto scale = static_cast<float>(std::sqrt(2)/avgDistance);

    for(int i = 0; i < static_cast<int>(points.size()); ++i) {
        norm_points[i] *= scale;
    }

    // Orginization of the normalization coefficient in a 3D
    // Transformation matrix
    Eigen::Matrix3f T;
    T << scale, 0, -scale * p1[0],
        0,      0, -scale * p1[1],
        0,      0,  1;

    return T;

}