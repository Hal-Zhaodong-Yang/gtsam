/* ----------------------------------------------------------------------------
 * GTSAM Copyright 2010, Georgia Tech Research Corporation,
 * Atlanta, Georgia 30332-0415
 * All Rights Reserved
 * Authors: Frank Dellaert, et al. (see THANKS for the full author list)
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

/**
 * @file    testTrifocalTensor2.cpp
 * @brief   Tests for the trifocal tensor class.
 * @author  Zhaodong Yang
 * @author  Akshay Krishnan
 */

#include <CppUnitLite/TestHarness.h>
#include <gtsam/base/Testable.h>
#include <gtsam/base/numericalDerivative.h>
#include <gtsam/geometry/BearingRange.h>
#include <gtsam/geometry/Pose2.h>
#include <gtsam/geometry/TrifocalTensor2.h>

#include <vector>

using namespace std::placeholders;
using namespace std;
using namespace gtsam;

namespace trifocal {

struct TrifocalTestData {
  vector<Pose2> gt_poses;
  TrifocalTensor2 gt_tensor;
  vector<Point2> gt_landmarks;

  // Outer vector over poses.
  std::vector<vector<Rot2>> measurements;
};

Rot2 bearingFromTranslation(const Point2& p) {
  Point2 atb = p / norm2(p);
  return Rot2::fromCosSin(atb[0], atb[1]);
}

TrifocalTestData getTestData() {
  TrifocalTestData data;

  // Poses
  data.gt_poses.emplace_back(0, 0, 0);
  data.gt_poses.emplace_back(-1.9, 4, -2 * acos(0.0) / 8);
  data.gt_poses.emplace_back(2.1, -2.1, 2 * acos(0.0) / 3);

  Rot2 aRb = data.gt_poses[1].rotation();
  Rot2 aRc = data.gt_poses[2].rotation();
  Rot2 atb = bearingFromTranslation(data.gt_poses[1].translation());
  Rot2 atc = bearingFromTranslation(data.gt_poses[2].translation());
  Rot2 btc = bearingFromTranslation(
      data.gt_poses[1].transformTo(data.gt_poses[2].translation()));
  data.gt_tensor = TrifocalTensor2(aRb, aRc, atb, atc, btc);

  // Landmarks
  data.gt_landmarks.emplace_back(1.2, 1.0);
  data.gt_landmarks.emplace_back(2.4, 3.5);
  data.gt_landmarks.emplace_back(-1.0, 0.5);
  data.gt_landmarks.emplace_back(3.4, -1.5);
  data.gt_landmarks.emplace_back(5.1, 0.6);
  data.gt_landmarks.emplace_back(-0.1, -0.7);
  data.gt_landmarks.emplace_back(3.1, 1.9);

  // Measurements
  for (const Pose2& pose : data.gt_poses) {
    std::vector<Rot2> measurements;
    for (const Point2& landmark : data.gt_landmarks) {
      measurements.push_back(pose.bearing(landmark));
    }
    data.measurements.push_back(measurements);
  }
  return data;
}

}  // namespace trifocal

// Check transform() correctly transforms measurements from 2 views to third.
TEST(TrifocalTensor2, transform) {
  trifocal::TrifocalTestData data = trifocal::getTestData();

  // calculate trifocal tensor
  TrifocalTensor2 T = TrifocalTensor2::FromBearingMeasurements(
      data.measurements[0], data.measurements[1], data.measurements[2]);

  // estimate measurement of a robot from the measurements of the other two
  // robots
  for (unsigned int i = 0; i < data.measurements[0].size(); i++) {
    const Rot2 actual_measurement =
        T.transform(data.measurements[1][i], data.measurements[2][i]);

    // there might be two solutions for u1 and u2, comparing the ratio instead
    // of both cos and sin
    EXPECT(assert_equal(actual_measurement.c() * data.measurements[0][i].s(),
                        actual_measurement.s() * data.measurements[0][i].c(),
                        1e-8));
  }
}

// Check the correct tensor is computed from measurements (catch regressions).
TEST(TrifocalTensor2, tensorRegression) {
  trifocal::TrifocalTestData data = trifocal::getTestData();
  TrifocalTensor2 test_tensor = trifocal::getTestData().gt_tensor;

  // calculate trifocal tensor
  TrifocalTensor2 T = TrifocalTensor2::FromBearingMeasurements(
      data.measurements[0], data.measurements[1], data.measurements[2]);

  Matrix2 expected_tensor_mat0, expected_tensor_mat1;

  expected_tensor_mat0 = test_tensor.mat0();
  expected_tensor_mat1 = test_tensor.mat1();

  Matrix2 actual_tensor_mat0 = T.mat0();
  Matrix2 actual_tensor_mat1 = T.mat1();

  double lambda = expected_tensor_mat0(0, 0) / actual_tensor_mat0(0, 0);

  actual_tensor_mat0 *= lambda;
  actual_tensor_mat1 *= lambda;

  EXPECT(assert_equal(expected_tensor_mat0, actual_tensor_mat0, 1e-2));
  EXPECT(assert_equal(expected_tensor_mat1, actual_tensor_mat1, 1e-2));
}

// Check the calculation of Jacobian (Ground-true Jacobian comes from Auto-Grad
// result of Pytorch)
// TEST(TrifocalTensor2, Jacobian) {
//   trifocal::TrifocalTestData data = trifocal::getTestData();

//   // Construct trifocal tensor using 2 rotations and 3 bearing measurements
//   in 3
//   // cameras.
//   std::vector<Rot2> trifocal_in_angle;
//   trifocal_in_angle.insert(
//       trifocal_in_angle.end(),
//       {-0.39269908169872414, 1.0471975511965976, 2.014244663214635,
//        -0.7853981633974483, -0.5976990577022983});

//   // calculate trifocal tensor
//   TrifocalTensor2 T(trifocal_in_angle);

//   // Calculate Jacobian matrix
//   Matrix jacobian_of_trifocal = T.Jacobian(
//       data.measurements[0], data.measurements[1], data.measurements[2]);
//   // These values were obtained from a Pytorch-based python implementation.
//   Matrix expected_jacobian(7, 5) << -2.2003, 0.7050, 0.9689, 0.6296, -3.1280,
//       -4.6886, 1.1274, 2.7912, 1.6121, -5.1817, -0.7223, -0.6841, 0.5387,
//       0.7208, -0.5677, -0.8645, 0.1767, 0.5967, 0.9383, -2.2041, -3.0437,
//       0.5239, 2.0144, 1.6368, -4.0335, -1.9855, -0.2741, 1.4741, 0.6783,
//       -0.9262, -4.6600, 0.7275, 2.8182, 1.9639, -5.5489;

//   EXPECT(assert_equal(jacobian_of_trifocal, expected_jacobian, 1e-8));
// }

// Testing equals() method.
TEST(TrifocalTensor2, equals) {
  TrifocalTensor2 test_tensor = trifocal::getTestData().gt_tensor;
  TrifocalTensor2 same_tensor = TrifocalTensor2(test_tensor);

  EXPECT(test_tensor.equals(same_tensor));  // same tensors are equal.
  EXPECT(!test_tensor.equals(
      TrifocalTensor2()));  // different tensors are unequal.
}

// Compute manifold representation from tensor, and convert back to tensor.
// Check whether result matches original.
TEST(TrifocalTensor2, minimalRepresentationRoundTrip) {
  TrifocalTensor2 test_tensor = trifocal::getTestData().gt_tensor;

  // Compute the manifold representation from the tensor.
  TrifocalTensor2 actual_tensor =
      TrifocalTensor2::FromTensor(test_tensor.mat0(), test_tensor.mat1());

  Matrix2 test_tensor_mat0 = test_tensor.mat0();
  Matrix2 test_tensor_mat1 = test_tensor.mat1();

  Matrix2 actual_tensor_mat0 = actual_tensor.mat0();
  Matrix2 actual_tensor_mat1 = actual_tensor.mat1();

  double lambda = test_tensor_mat0(0, 0) / actual_tensor_mat0(0, 0);

  actual_tensor_mat0 *= lambda;
  actual_tensor_mat1 *= lambda;

  // Get the tensor back from the manifold representation, comapre to original.
  EXPECT(assert_equal(test_tensor_mat0, actual_tensor_mat0));
  EXPECT(assert_equal(test_tensor_mat1, actual_tensor_mat1));
}

// Test jacobian of FromTensor().

TEST(TrifocalTensor2, FromTensorJacobians) {
  TrifocalTensor2 test_tensor = trifocal::getTestData().gt_tensor;
  std::function<TrifocalTensor2(const Matrix2&, const Matrix2&)> f =
      std::bind(&TrifocalTensor2::FromTensor, std::placeholders::_1,
                std::placeholders::_2, boost::none);

  Matrix54 expectedH1 =
      numericalDerivative21(f, test_tensor.mat0(), test_tensor.mat1());
  Matrix54 expectedH2 =
      numericalDerivative22(f, test_tensor.mat0(), test_tensor.mat1());

  Matrix58 actual_H;
  TrifocalTensor2 result = TrifocalTensor2::FromTensor(
      test_tensor.mat0(), test_tensor.mat1(), actual_H);
  EXPECT(assert_equal<Matrix54>(expectedH1, actual_H.topLeftCorner(5, 4)));
  EXPECT(assert_equal<Matrix54>(expectedH2, actual_H.topRightCorner(5, 4)));
}


Rot2 tranformBearing(const TrifocalTensor2& tensor, const Rot2& theta_b,
                     const Rot2& theta_c) {
  return tensor.transform(theta_b, theta_c);
}

// test jacobian of transform()

TEST(TrifocalTensor2, transformJacobian) {
  trifocal::TrifocalTestData data = trifocal::getTestData();
  Rot2 theta_b = data.measurements[0][0], theta_c = data.measurements[1][0];
  std::function<Rot2(const TrifocalTensor2&)> f =
      std::bind(&tranformBearing, std::placeholders::_1, theta_b, theta_c);

  Matrix15 expected_H = numericalDerivative11(f, data.gt_tensor);

  Matrix15 actual_H;
  Rot2 result = data.gt_tensor.transform(theta_b, theta_c, actual_H);
  EXPECT(assert_equal(expected_H, actual_H));
}


TEST(TrifocalTensor2, tensorConversionJacobian) {
  TrifocalTensor2 test_tensor = trifocal::getTestData().gt_tensor;

  // First matrix

  std::function<Matrix2(const TrifocalTensor2&)> f0 =
      std::bind(&TrifocalTensor2::mat0, std::placeholders::_1, boost::none);
  Matrix45 expected_H0 = numericalDerivative11(f0, test_tensor);

  Matrix45 actual_H0;
  Matrix2 result0 = test_tensor.mat0(actual_H0);
  EXPECT(assert_equal(expected_H0, actual_H0));

  // Second matrix
  std::function<Matrix2(const TrifocalTensor2&)> f1 =
      std::bind(&TrifocalTensor2::mat1, std::placeholders::_1, boost::none);

  Matrix45 expected_H1 = numericalDerivative11(f1, test_tensor);
  Matrix45 actual_H1;
  Matrix2 result1 = test_tensor.mat1(actual_H1);
  EXPECT(assert_equal(expected_H1, actual_H1));
}

int main() {
  TestResult tr;
  return TestRegistry::runAllTests(tr);
}
