/* ----------------------------------------------------------------------------
 * GTSAM Copyright 2010, Georgia Tech Research Corporation,
 * Atlanta, Georgia 30332-0415
 * All Rights Reserved
 * Authors: Frank Dellaert, et al. (see THANKS for the full author list)
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

/**
 * @file    TrifocalTensor2.cpp
 * @brief   A 2x2x2 trifocal tensor in a plane, for 1D cameras.
 * @author  Zhaodong Yang
 * @author  Akshay Krishnan
 */

#include <gtsam/geometry/Pose2.h>
#include <gtsam/geometry/TrifocalTensor2.h>

#include <stdexcept>
#include <vector>

namespace gtsam {

// Convert bearing measurements to projective coordinates.
std::vector<Point2> convertToProjective(const std::vector<Rot2>& rotations) {
  std::vector<Point2> projectives;
  projectives.reserve(rotations.size());
  for (const Rot2& rotation : rotations) {
    projectives.emplace_back(rotation.c() / rotation.s(), 1.0);
  }
  return projectives;
}

std::pair<Pose2, Pose2> posesFromMinimal(
    const Rot2& aRb, const Rot2& aRc, const Rot2& atb, const Rot2& atc,
    const Rot2& btc, OptionalJacobian<6, 5> H = boost::none) {
  // TODO: this notation is not correct: bTw should accept bRa, change to wTb
  // and use inverse().
  Pose2 bTw(aRb, Point2(-cos(atb.theta() - aRb.theta()),
                        -sin(atb.theta() - aRb.theta())));
  double lambda = sin(btc.theta() + aRb.theta() - atb.theta()) /
                  sin(btc.theta() + aRb.theta() - atc.theta());
  Pose2 cTw(aRc, -1 * lambda *
                     Point2(cos(atc.theta() - aRc.theta()),
                            sin(atc.theta() - aRc.theta())));
  return std::make_pair(bTw, cTw);
}

std::pair<Matrix2, Matrix2> tensorFromPose(const Pose2& bTw, const Pose2& cTw) {
  Point2 ctw = cTw.translation(), btw = bTw.translation();
  // TODO: another notational incosistency here.
  Rot2 wRb = bTw.rotation(), wRc = cTw.rotation();
  Matrix2 mat0, mat1;
  mat0(0, 0) = ctw.y() * wRb.s() - btw.y() * wRc.s();
  mat0(0, 1) = -ctw.x() * wRb.s() - btw.y() * wRc.c();
  mat0(1, 0) = ctw.y() * wRb.c() + btw.x() * wRc.s();
  mat0(1, 1) = -ctw.x() * wRb.c() + btw.x() * wRc.c();
  mat1(0, 0) = -ctw.y() * wRb.c() + btw.y() * wRc.c();
  mat1(0, 1) = ctw.x() * wRb.c() - btw.y() * wRc.s();
  mat1(1, 0) = ctw.y() * wRb.s() - btw.x() * wRc.c();
  mat1(1, 1) = -ctw.x() * wRb.s() + btw.x() * wRc.s();
  return std::make_pair(mat0, mat1);
}

// this function is intermediate calculation of trifocal matrix from trifocal
// tensor in minimal representation. The returning value is 9 variables which
// can be used to calculate trifocal matrix, and the 9 variables can be
// calculated from 5 angles, which are the minimal representation of trifocal
// tensor
Vector9 intermediaFromMinimal(const Rot2& theta_prime,
                              const Rot2& theta_double_prime,
                              const Rot2& theta1, const Rot2& theta2,
                              const Rot2& theta3,
                              OptionalJacobian<9, 5> Dtensor) {
  Vector9 intermedia_function;
  Matrix jacobian(9, 5);
  // 9 variables which appear in formula of trifocal matrix repeatedly
  intermedia_function(0) = sin(theta2.theta() - theta_double_prime.theta());
  intermedia_function(1) = cos(theta2.theta() - theta_double_prime.theta());

  intermedia_function(2) = theta_prime.s();
  intermedia_function(3) = theta_prime.c();

  intermedia_function(4) = sin(-theta_prime.theta() + theta1.theta());
  intermedia_function(5) = cos(-theta_prime.theta() + theta1.theta());

  intermedia_function(6) = theta_double_prime.s();
  intermedia_function(7) = theta_double_prime.c();

  intermedia_function(8) =
      sin(theta3.theta() + theta_prime.theta() - theta1.theta()) /
      sin(theta3.theta() + theta_prime.theta() - theta2.theta());

  // jacobian of the 9 variables wrt 5 angles (minimal representation of
  // trifocal tensor)
  if (Dtensor) {
    for (int i = 0; i < 9; i++) {
      for (int j = 0; j < 5; j++) {
        jacobian(i, j) = 0;
      }
    }
    jacobian(0, 1) = -cos(theta2.theta() - theta_double_prime.theta());
    jacobian(1, 1) = sin(theta2.theta() - theta_double_prime.theta());
    jacobian(0, 3) = cos(theta2.theta() - theta_double_prime.theta());
    jacobian(1, 3) = -sin(theta2.theta() - theta_double_prime.theta());

    jacobian(2, 0) = theta_prime.c();
    jacobian(3, 0) = -theta_prime.s();

    jacobian(4, 0) = -cos(-theta_prime.theta() + theta1.theta());
    jacobian(5, 0) = sin(-theta_prime.theta() + theta1.theta());
    jacobian(4, 2) = cos(-theta_prime.theta() + theta1.theta());
    jacobian(5, 2) = -sin(-theta_prime.theta() + theta1.theta());

    jacobian(6, 1) = theta_double_prime.c();
    jacobian(7, 1) = -theta_double_prime.s();

    jacobian(8, 0) =
        (cos(theta3.theta() + theta_prime.theta() - theta1.theta()) *
             sin(theta3.theta() + theta_prime.theta() - theta2.theta()) -
         sin(theta3.theta() + theta_prime.theta() - theta1.theta()) *
             cos(theta3.theta() + theta_prime.theta() - theta2.theta())) /
        (sin(theta3.theta() + theta_prime.theta() - theta2.theta()) *
         sin(theta3.theta() + theta_prime.theta() - theta2.theta()));

    jacobian(8, 2) =
        -cos(theta3.theta() + theta_prime.theta() - theta1.theta()) /
        sin(theta3.theta() + theta_prime.theta() - theta2.theta());

    jacobian(8, 3) =
        sin(theta3.theta() + theta_prime.theta() - theta1.theta()) *
        cos(theta3.theta() + theta_prime.theta() - theta2.theta()) /
        (sin(theta3.theta() + theta_prime.theta() - theta2.theta()) *
         sin(theta3.theta() + theta_prime.theta() - theta2.theta()));

    jacobian(8, 4) = jacobian(8, 0);

    *Dtensor << jacobian;
  }

  return intermedia_function;
}

Rot2 retractWithRot2(const Rot2& r, const Vector5& v, int idx,
                     OptionalJacobian<5, 5> Dtensor,
                     OptionalJacobian<5, 5> Dv) {
  Matrix1 Drot2, Dv1;
  Vector1 v1(v[idx]);
  Rot2 result = r.retract(v1, Drot2, Dv1);
  if (Dtensor) (*Dtensor)(idx, idx) = Drot2(0, 0);
  if (Dv) (*Dv)(idx, idx) = Dv1(0, 0);
  return result;
}

Vector1 localCoordinatesRot2(const Rot2& this_rot, const Rot2& other, int idx,
                             OptionalJacobian<5, 5> Dtensor,
                             OptionalJacobian<5, 5> Dother) {
  Matrix1 Dthis, Dother1;
  Vector1 result = this_rot.localCoordinates(other, Dthis, Dother1);
  if (Dtensor) (*Dtensor)(idx, idx) = Dthis(0, 0);
  if (Dother) (*Dother)(idx, idx) = Dother1(0, 0);
  return result;
}

// Construct from 8 bearing measurements.
TrifocalTensor2 TrifocalTensor2::FromBearingMeasurements(
    const std::vector<Rot2>& bearings_u, const std::vector<Rot2>& bearings_v,
    const std::vector<Rot2>& bearings_w) {
  return TrifocalTensor2::FromProjectiveBearingMeasurements(
      convertToProjective(bearings_u), convertToProjective(bearings_v),
      convertToProjective(bearings_w));
}

// Construct from 8 bearing measurements expressed in projective coordinates.
TrifocalTensor2 TrifocalTensor2::FromProjectiveBearingMeasurements(
    const std::vector<Point2>& u, const std::vector<Point2>& v,
    const std::vector<Point2>& w) {
  if (u.size() < 7) {
    throw std::invalid_argument(
        "Trifocal tensor computation requires at least 8 measurements");
  }
  if (u.size() != v.size() || v.size() != w.size()) {
    throw std::invalid_argument(
        "Number of input measurements in 3 cameras must be same");
  }

  // Create the system matrix A.
  Matrix A(u.size() > 8 ? u.size() : 8, 8);
  for (int row = 0; row < u.size(); row++) {
    for (int i = 0; i < 2; i++) {
      for (int j = 0; j < 2; j++) {
        for (int k = 0; k < 2; k++) {
          A(row, 4 * i + 2 * j + k) = u[row](i) * v[row](j) * w[row](k);
        }
      }
    }
  }
  for (int row = u.size(); row < 8; row++) {
    for (int col = 0; col < 8; col++) {
      A(row, col) = 0;
    }
  }

  // Eigen vector of smallest singular value is the trifocal tensor.
  Matrix U, V;
  Vector S;
  svd(A, U, S, V);

  Matrix2 matrix0, matrix1;
  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 2; j++) {
      matrix0(i, j) = V(2 * i + j, V.cols() - 1);
      matrix1(i, j) = V(2 * i + j + 4, V.cols() - 1);
    }
  }
  return TrifocalTensor2::FromTensor(matrix0, matrix1);
}

TrifocalTensor2 TrifocalTensor2::FromTensor(const Matrix2& matrix0,
                                            const Matrix2& matrix1,
                                            OptionalJacobian<5, 8> Dtensor) {
  // KCB is the homography transformation from view2 to view3, LCB is the
  // homography transformation from view3 to view2
  Matrix2 KCB;
  KCB << -matrix0(0, 1), -matrix0(1, 1), matrix0(0, 0), matrix0(1, 0);
  Matrix2 LCB;
  LCB << -matrix1(0, 1), -matrix1(1, 1), matrix1(0, 0), matrix1(1, 0);
  Matrix2 KAB;
  KAB << -matrix1(0, 0), -matrix1(1, 0), matrix0(0, 0), matrix0(1, 0);

  // M is the homography transformation from view2 to itself
  Matrix2 M = LCB.inverse() * KCB;

  // this part is to calculate the eigenvector of M
  double trace_M = M(0, 0) + M(1, 1);
  double det_M = M(0, 0) * M(1, 1) - M(0, 1) * M(1, 0);
  double eigenvalue1 = (trace_M + sqrt(trace_M * trace_M - 4 * det_M)) / 2.0;
  double eigenvalue2 = (trace_M - sqrt(trace_M * trace_M - 4 * det_M)) / 2.0;

  // the eigenvector, which is also epipoles of view1 and view3 in view2
  // TODO @Hal-Zhaodong-Yang: There's multiplicity of solution. Eigenvector can
  // be switched
  Vector2 epipoleB1;
  epipoleB1 << -M(0, 1), M(0, 0) - eigenvalue1;
  Vector2 epipoleB3;
  epipoleB3 << -M(0, 1), M(0, 0) - eigenvalue2;

  // calculate epipoles in other views by projecting back
  Vector2 epipoleA2 = KAB * epipoleB1;
  Vector2 epipoleC2 = KCB * epipoleB3;
  Vector2 epipoleA3 = KAB * epipoleB3;
  Vector2 epipoleC1 = KCB * epipoleB1;

  Rot2 aRb, aRc, atb, atc, btc;
  // TODO @Hal-Zhaodong-Yang: transformation from projection to bearing is not a
  // surjection. There's multiplicity of solution
  atb = Rot2::atan2(epipoleA2(1), epipoleA2(0));
  atc = Rot2::atan2(epipoleA3(1), epipoleA3(0));
  btc = Rot2::atan2(epipoleB3(1), epipoleB3(0));
  aRb = Rot2(atan2(epipoleA2(1), epipoleA2(0)) -
             atan2(epipoleB1(1), epipoleB1(0)));
  aRc = Rot2(atan2(epipoleA3(1), epipoleA3(0)) -
             atan2(epipoleC1(1), epipoleC1(0)));

  return TrifocalTensor2(aRb, aRc, atb, atc, btc);
}

// Finds a measurement in the first view using measurements from second and
// third views.
Rot2 TrifocalTensor2::transform(const Rot2& vZp, const Rot2& wZp,
                                OptionalJacobian<1, 5> Dtensor) const {
  Rot2 uZp;
  Vector2 v_measurement, w_measurement;
  v_measurement << vZp.c(), vZp.s();
  w_measurement << wZp.c(), wZp.s();
  return Rot2::atan2(dot(mat0() * w_measurement, v_measurement),
                     -dot(mat1() * w_measurement, v_measurement));
}

Matrix2 TrifocalTensor2::mat0(OptionalJacobian<4, 5> Dtensor) const {
  Matrix2 matrix0;
  Matrix Dintermedia_wrt_minimal(9, 5);

  Vector9 intermedia = intermediaFromMinimal(aRb_, aRc_, atb_, atc_, btc_,
                                             Dintermedia_wrt_minimal);

  // trifocal matrix formula using intermediate variables
  matrix0(0, 0) = -intermedia(8) * intermedia(0) * intermedia(2) +
                  intermedia(4) * intermedia(6);
  matrix0(0, 1) = intermedia(8) * intermedia(1) * intermedia(2) +
                  intermedia(4) * intermedia(7);
  matrix0(1, 0) = -intermedia(8) * intermedia(0) * intermedia(3) -
                  intermedia(5) * intermedia(6);
  matrix0(1, 1) = intermedia(8) * intermedia(1) * intermedia(3) -
                  intermedia(5) * intermedia(7);

  if (Dtensor) {
    Matrix Dtensor_wrt_intermedia(4, 9);
    for (int i = 0; i < 4; i++) {
      for (int j = 0; j < 9; j++) {
        Dtensor_wrt_intermedia(i, j) = 0;
      }
    }
    // calculate jacobian according to the formula above
    Dtensor_wrt_intermedia(0, 0) = -intermedia(8) * intermedia(2);
    Dtensor_wrt_intermedia(0, 2) = -intermedia(8) * intermedia(0);
    Dtensor_wrt_intermedia(0, 8) = -intermedia(0) * intermedia(2);
    Dtensor_wrt_intermedia(0, 4) = intermedia(6);
    Dtensor_wrt_intermedia(0, 6) = intermedia(4);

    Dtensor_wrt_intermedia(1, 1) = intermedia(8) * intermedia(2);
    Dtensor_wrt_intermedia(1, 2) = intermedia(8) * intermedia(1);
    Dtensor_wrt_intermedia(1, 8) = intermedia(1) * intermedia(2);
    Dtensor_wrt_intermedia(1, 4) = intermedia(7);
    Dtensor_wrt_intermedia(1, 7) = intermedia(4);

    Dtensor_wrt_intermedia(2, 0) = -intermedia(8) * intermedia(3);
    Dtensor_wrt_intermedia(2, 3) = -intermedia(8) * intermedia(0);
    Dtensor_wrt_intermedia(2, 8) = -intermedia(0) * intermedia(3);
    Dtensor_wrt_intermedia(2, 5) = -intermedia(6);
    Dtensor_wrt_intermedia(2, 6) = -intermedia(5);

    Dtensor_wrt_intermedia(3, 1) = intermedia(8) * intermedia(3);
    Dtensor_wrt_intermedia(3, 3) = intermedia(8) * intermedia(1);
    Dtensor_wrt_intermedia(3, 8) = intermedia(1) * intermedia(3);
    Dtensor_wrt_intermedia(3, 5) = -intermedia(7);
    Dtensor_wrt_intermedia(3, 7) = -intermedia(5);

    *Dtensor << Dtensor_wrt_intermedia * Dintermedia_wrt_minimal;
  }
  auto pose_pair = posesFromMinimal(aRb_, aRc_, atb_, atc_, btc_);
  auto mat_pair = tensorFromPose(pose_pair.first, pose_pair.second);
  return mat_pair.first;
}

Matrix2 TrifocalTensor2::mat1(OptionalJacobian<4, 5> Dtensor) const {
  Matrix2 matrix1;
  Matrix Dintermedia_wrt_minimal(9, 5);

  Vector9 intermedia = intermediaFromMinimal(aRb_, aRc_, atb_, atc_, btc_,
                                             &Dintermedia_wrt_minimal);

  // trifocal matrix formula using intermediate variables (different with mat0)
  matrix1(0, 0) = intermedia(8) * intermedia(0) * intermedia(3) -
                  intermedia(4) * intermedia(7);
  matrix1(0, 1) = -intermedia(8) * intermedia(1) * intermedia(3) +
                  intermedia(4) * intermedia(6);
  matrix1(1, 0) = -intermedia(8) * intermedia(0) * intermedia(2) +
                  intermedia(5) * intermedia(7);
  matrix1(1, 1) = intermedia(8) * intermedia(1) * intermedia(2) -
                  intermedia(5) * intermedia(6);

  if (Dtensor) {
    Matrix Dtensor_wrt_intermedia(4, 9);
    for (int i = 0; i < 4; i++) {
      for (int j = 0; j < 9; j++) {
        Dtensor_wrt_intermedia(i, j) = 0;
      }
    }
    // calculate jacobian according to the formula above
    Dtensor_wrt_intermedia(0, 0) = intermedia(8) * intermedia(3);
    Dtensor_wrt_intermedia(0, 3) = intermedia(8) * intermedia(0);
    Dtensor_wrt_intermedia(0, 8) = intermedia(0) * intermedia(3);
    Dtensor_wrt_intermedia(0, 4) = -intermedia(7);
    Dtensor_wrt_intermedia(0, 7) = -intermedia(4);

    Dtensor_wrt_intermedia(1, 1) = -intermedia(8) * intermedia(3);
    Dtensor_wrt_intermedia(1, 3) = -intermedia(8) * intermedia(1);
    Dtensor_wrt_intermedia(1, 8) = -intermedia(1) * intermedia(3);
    Dtensor_wrt_intermedia(1, 4) = intermedia(6);
    Dtensor_wrt_intermedia(1, 6) = intermedia(4);

    Dtensor_wrt_intermedia(2, 0) = -intermedia(8) * intermedia(2);
    Dtensor_wrt_intermedia(2, 2) = -intermedia(8) * intermedia(0);
    Dtensor_wrt_intermedia(2, 8) = -intermedia(0) * intermedia(2);
    Dtensor_wrt_intermedia(2, 5) = intermedia(7);
    Dtensor_wrt_intermedia(2, 7) = intermedia(5);

    Dtensor_wrt_intermedia(3, 1) = intermedia(8) * intermedia(2);
    Dtensor_wrt_intermedia(3, 2) = intermedia(8) * intermedia(1);
    Dtensor_wrt_intermedia(3, 8) = intermedia(1) * intermedia(2);
    Dtensor_wrt_intermedia(3, 5) = -intermedia(6);
    Dtensor_wrt_intermedia(3, 6) = -intermedia(5);

    *Dtensor << Dtensor_wrt_intermedia * Dintermedia_wrt_minimal;
  }

  auto pose_pair = posesFromMinimal(aRb_, aRc_, atb_, atc_, btc_);
  auto mat_pair = tensorFromPose(pose_pair.first, pose_pair.second);
  return mat_pair.second;
}

void TrifocalTensor2::print(const std::string& s) const {
  std::cout << s << std::endl;
  std::cout << "aRb: " << aRb_.theta() << ", aRc: " << aRc_.theta()
            << ", atb: " << atb_.theta() << ", atc:" << atc_.theta()
            << ", btc: " << btc_.theta() << std::endl;
}

TrifocalTensor2 TrifocalTensor2::retract(const Vector5& v,
                                         OptionalJacobian<5, 5> Dtensor,
                                         OptionalJacobian<5, 5> Dv) const {
  if (Dtensor) (*Dtensor).setZero();
  if (Dv) (*Dv).setZero();
  Rot2 aRb_sum = retractWithRot2(aRb_, v, 0, Dtensor, Dv);
  Rot2 aRc_sum = retractWithRot2(aRc_, v, 1, Dtensor, Dv);
  Rot2 atb_sum = retractWithRot2(atb_, v, 2, Dtensor, Dv);
  Rot2 atc_sum = retractWithRot2(atc_, v, 3, Dtensor, Dv);
  Rot2 btc_sum = retractWithRot2(btc_, v, 4, Dtensor, Dv);
  return TrifocalTensor2(aRb_sum, aRc_sum, atb_sum, atc_sum, btc_sum);
}

Vector5 TrifocalTensor2::localCoordinates(const TrifocalTensor2& other,
                                          OptionalJacobian<5, 5> Dtensor,
                                          OptionalJacobian<5, 5> Dother) const {
  if (Dtensor) (*Dtensor).setZero();
  if (Dother) (*Dother).setZero();
  Vector1 aRb_diff =
      localCoordinatesRot2(aRb_, other.aRb(), 0, Dtensor, Dother);
  Vector1 aRc_diff =
      localCoordinatesRot2(aRc_, other.aRc(), 1, Dtensor, Dother);
  Vector1 atb_diff =
      localCoordinatesRot2(atb_, other.atb(), 2, Dtensor, Dother);
  Vector1 atc_diff =
      localCoordinatesRot2(atc_, other.atc(), 3, Dtensor, Dother);
  Vector1 btc_diff =
      localCoordinatesRot2(btc_, other.btc(), 4, Dtensor, Dother);
  Vector5 result;
  result << aRb_diff[0], aRc_diff[0], atb_diff[0], atc_diff[0], btc_diff[0];
  return result;
}

bool TrifocalTensor2::equals(const TrifocalTensor2& other, double tol) const {
  return aRb_.equals(other.aRb(), tol) && aRc_.equals(other.aRc(), tol) &&
         atb_.equals(other.atb(), tol) && atc_.equals(other.atc(), tol) &&
         btc_.equals(other.btc(), tol);
}

}  // namespace gtsam