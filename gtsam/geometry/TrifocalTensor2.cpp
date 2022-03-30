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

double calTrifocalFromIntermedia(double x1, double x2, double x3, double x4,
                                 double x5) {
  return x1 * x2 * x3 + x4 * x5;
}

Matrix89 computeJacobianWrtIntermedia(const Vector9& f) {
  Matrix89 H;
  H.row(0) << -f(8) * f(2), 0, -f(8) * f(0), 0, f(6), 0, f(4), 0, -f(0) * f(2);
  H.row(1) << 0, f(8) * f(2), f(8) * f(1), 0, f(7), 0, 0, f(4), f(1) * f(2);
  H.row(2) << -f(8) * f(3), 0, 0, -f(8) * f(0), 0, -f(6), -f(5), 0,
      -f(0) * f(3);
  H.row(3) << 0, f(8) * f(3), 0, f(8) * f(1), 0, -f(7), 0, -f(5), f(1) * f(3);
  H.row(4) << f(8) * f(3), 0, 0, f(8) * f(0), -f(7), 0, 0, -f(4), f(0) * f(3);
  H.row(5) << 0, -f(8) * f(3), 0, -f(8) * f(1), f(6), 0, f(4), 0, -f(1) * f(3);
  H.row(6) << -f(8) * f(2), 0, -f(8) * f(0), 0, 0, f(7), 0, f(5), -f(0) * f(2);
  H.row(7) << 0, f(8) * f(2), f(8) * f(1), 0, 0, -f(6), -f(5), 0, f(1) * f(2);
  return H;
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
  Vector9 f;
  Matrix jacobian(9, 5);
  // 9 variables which appear in formula of trifocal matrix repeatedly
  f(0) = (theta2 * theta_double_prime.inverse()).s();
  f(1) = (theta2 * theta_double_prime.inverse()).c();

  f(2) = theta_prime.s();
  f(3) = theta_prime.c();

  f(4) = (theta_prime.inverse() * theta1).s();
  f(5) = (theta_prime.inverse() * theta1).c();

  f(6) = theta_double_prime.s();
  f(7) = theta_double_prime.c();

  f(8) = (theta3 * theta_prime * theta1.inverse()).s() /
         (theta3 * theta_prime * theta2.inverse()).s();

  // jacobian of the 9 variables wrt 5 angles (minimal representation of
  // trifocal tensor)
  if (Dtensor) {
    for (int i = 0; i < 9; i++) {
      for (int j = 0; j < 5; j++) {
        jacobian(i, j) = 0;
      }
    }
    jacobian(0, 1) = -(theta2 * theta_double_prime.inverse()).c();
    jacobian(1, 1) = (theta2 * theta_double_prime.inverse()).s();
    jacobian(0, 3) = (theta2 * theta_double_prime.inverse()).c();
    jacobian(1, 3) = -(theta2 * theta_double_prime.inverse()).s();

    jacobian(2, 0) = theta_prime.c();
    jacobian(3, 0) = -theta_prime.s();

    jacobian(4, 0) = -(theta_prime.inverse() * theta1).c();
    jacobian(5, 0) = (theta_prime.inverse() * theta1).s();
    jacobian(4, 2) = (theta_prime.inverse() * theta1).c();
    jacobian(5, 2) = -(theta_prime.inverse() * theta1).s();

    jacobian(6, 1) = theta_double_prime.c();
    jacobian(7, 1) = -theta_double_prime.s();

    jacobian(8, 0) = ((theta3 * theta_prime * theta1.inverse()).c() *
                          (theta3 * theta_prime * theta2.inverse()).s() -
                      (theta3 * theta_prime * theta1.inverse()).s() *
                          (theta3 * theta_prime * theta2.inverse()).c()) /
                     ((theta3 * theta_prime * theta2.inverse()).s() *
                      (theta3 * theta_prime * theta2.inverse()).s());

    jacobian(8, 2) = -(theta3 * theta_prime * theta1.inverse()).c() /
                     (theta3 * theta_prime * theta2.inverse()).s();

    jacobian(8, 3) = (theta3 * theta_prime * theta1.inverse()).s() *
                     (theta3 * theta_prime * theta2.inverse()).c() /
                     ((theta3 * theta_prime * theta2.inverse()).s() *
                      (theta3 * theta_prime * theta2.inverse()).s());

    jacobian(8, 4) = jacobian(8, 0);

    *Dtensor << jacobian;
  }

  return f;
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
    const std::vector<Rot2>& bearings_a, const std::vector<Rot2>& bearings_b,
    const std::vector<Rot2>& bearings_c) {
  return TrifocalTensor2::FromProjectiveBearingMeasurements(
      convertToProjective(bearings_a), convertToProjective(bearings_b),
      convertToProjective(bearings_c));
}

// Construct from 8 bearing measurements expressed in projective coordinates.
TrifocalTensor2 TrifocalTensor2::FromProjectiveBearingMeasurements(
    const std::vector<Point2>& a, const std::vector<Point2>& b,
    const std::vector<Point2>& c) {
  if (a.size() < 7) {
    throw std::invalid_argument(
        "Trifocal tensor computation requires at least 7 measurements");
  }
  if (a.size() != b.size() || b.size() != c.size()) {
    throw std::invalid_argument(
        "Number of input measurements in 3 cameras must be same");
  }

  // Create the system matrix A.
  Matrix A(a.size() > 8 ? a.size() : 8, 8);
  for (int row = 0; row < a.size(); row++) {
    for (int i = 0; i < 2; i++) {
      for (int j = 0; j < 2; j++) {
        for (int k = 0; k < 2; k++) {
          A(row, 4 * i + 2 * j + k) = a[row](i) * b[row](j) * c[row](k);
        }
      }
    }
  }
  for (int row = a.size(); row < 8; row++) {
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
  Vector2 epipoleB3;
  epipoleB3 << -M(0, 1), M(0, 0) - eigenvalue1;
  epipoleB1 << -M(0, 1), M(0, 0) - eigenvalue2;

  // calculate epipoles in other views by projecting back
  Vector2 epipoleA2 = KAB * epipoleB1;
  Vector2 epipoleC2 = KCB * epipoleB3;
  Vector2 epipoleA3 = KAB * epipoleB3;
  Vector2 epipoleC1 = KCB * epipoleB1;

  Rot2 aRb, aRc, atb, atc, btc;
  // TODO @Hal-Zhaodong-Yang: transformation from projection to bearing is not a
  // surjection. There's multiplicity of solution. Add constraint to check if the
  // solution is self-consistent
  atb = Rot2::atan2(epipoleA2(1), epipoleA2(0));
  atc = Rot2::atan2(epipoleA3(1), epipoleA3(0));
  btc = Rot2::atan2(epipoleB3(1), epipoleB3(0));
  aRb = Rot2(atan2(epipoleA2(1), epipoleA2(0)) -
             atan2(epipoleB1(1), epipoleB1(0)));
  aRc = Rot2(atan2(epipoleA3(1), epipoleA3(0)) -
             atan2(epipoleC1(1), epipoleC1(0)));
  // change the initialized tensor to (0, pi)
  Rot2 pi = Rot2::fromCosSin(-1, 0);
  if (atb.theta() < 0){
    atb = atb * pi;
  }
  if (atc.theta() < 0){
    atc = atc * pi;
  }
  if (btc.theta() < 0){
    btc = btc * pi;
  }
  if (aRb.theta() < 0){
    aRb = aRb * pi;
  }
  if (aRc.theta() < 0){
    aRc = aRc * pi;
  }

  return TrifocalTensor2(aRb, aRc, atb, atc, btc);
}

// Finds a measurement in the first view using measurements from second and
// third views.
Rot2 TrifocalTensor2::transform(const Rot2& bZp, const Rot2& cZp,
                                OptionalJacobian<1, 5> Dtensor) const {
  Vector2 bp = bZp.rotate(Point2(1, 0));
  Vector2 cp = cZp.rotate(Point2(1, 0));
  Matrix25 Dap_tensor;
  Point2 ap = transform(bp, cp, Dap_tensor);
  Matrix12 DaZp_ap;
  Rot2 aZp = Rot2::relativeBearing(ap, DaZp_ap);
  if(Dtensor) {
    *Dtensor = DaZp_ap * Dap_tensor;
  }
  return aZp;
}

Point2 TrifocalTensor2::transform(const Point2& bp, const Point2& cp,
                                  OptionalJacobian<2, 5> Dtensor) const {
  Matrix85 Dfull_tensor;
  const auto t = tensor(Dfull_tensor);
  if (Dtensor) {
    Matrix28 Dap_tensor;
    Dap_tensor << 0.0, 0.0, 0.0, 0.0, -bp(0) * cp(0), -bp(0) * cp(1),
        -bp(1) * cp(0), -bp(1) * cp(1), bp(0) * cp(0), bp(0) * cp(1),
        bp(1) * cp(0), bp(1) * cp(1), 0.0, 0.0, 0.0, 0.0;
    *Dtensor = Dap_tensor * Dfull_tensor;
  }
  return Point2(-dot(t.second * cp, bp), dot(t.first * cp, bp));
}

std::pair<Matrix2, Matrix2> TrifocalTensor2::tensor(
    OptionalJacobian<8, 5> Dtensor) const {
  Matrix2 matrix0, matrix1;
  Matrix Dintermedia_wrt_minimal(9, 5);
  Vector9 f = intermediaFromMinimal(aRb_, aRc_, atb_, atc_, btc_,
                                    Dintermedia_wrt_minimal);

  // trifocal matrix formula using intermediate variables
  matrix0(0, 0) = calTrifocalFromIntermedia(-f(8), f(0), f(2), f(4), f(6));
  matrix0(0, 1) = calTrifocalFromIntermedia(f(8), f(1), f(2), f(4), f(7));
  matrix0(1, 0) = calTrifocalFromIntermedia(-f(8), f(0), f(3), -f(5), f(6));
  matrix0(1, 1) = calTrifocalFromIntermedia(f(8), f(1), f(3), -f(5), f(7));
  matrix1(0, 0) = calTrifocalFromIntermedia(f(8), f(0), f(3), -f(4), f(7));
  matrix1(0, 1) = calTrifocalFromIntermedia(-f(8), f(1), f(3), f(4), f(6));
  matrix1(1, 0) = calTrifocalFromIntermedia(-f(8), f(0), f(2), f(5), f(7));
  matrix1(1, 1) = calTrifocalFromIntermedia(f(8), f(1), f(2), -f(5), f(6));

  if (Dtensor) {
    Matrix89 Dtensor_wrt_intermedia = computeJacobianWrtIntermedia(f);
    *Dtensor << Dtensor_wrt_intermedia * Dintermedia_wrt_minimal;
  }

  return std::make_pair(matrix0, matrix1);
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