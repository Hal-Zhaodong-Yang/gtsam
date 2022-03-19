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
 * @author  Akshay Krishnan
 * @author  Zhaodong Yang
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

// Creates a test dataset.
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

// Wrapper function for unit test of tensorConversionJacobian.
Vector8 tensorPairToVector(const TrifocalTensor2& tensor) {
  Vector8 trifocal_tensor;
  std::pair<Matrix2, Matrix2> t = tensor.tensor();
  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 2; j++) {
      trifocal_tensor(2 * i + j) = t.first(i, j);
      trifocal_tensor(2 * i + j + 4) = t.second(i, j);
    }
  }
  return trifocal_tensor;
}

}  // namespace trifocal

// Check transform() correctly transforms measurements from views B, C to A.
TEST(TrifocalTensor2, transform) {
  trifocal::TrifocalTestData data = trifocal::getTestData();

  // Compute trifocal tensor
  TrifocalTensor2 T = TrifocalTensor2::FromBearingMeasurements(
      data.measurements[0], data.measurements[1], data.measurements[2]);

  // Estimate measurement in view A from measurements in B and C
  for (unsigned int i = 0; i < data.measurements[0].size(); i++) {
    const Rot2 actual_measurement =
        T.transform(data.measurements[1][i], data.measurements[2][i]);

    // TODO(zhaodong): how do we fix this?
    // there might be two solutions for u1 and u2, comparing the ratio instead
    // of both cos and sin
    EXPECT(assert_equal(actual_measurement.c() * data.measurements[0][i].s(),
                        actual_measurement.s() * data.measurements[0][i].c(),
                        1e-8));
  }
}

// Check the correct tensor is computed from measurements (catch regressions).
// The tensor is computed only upto a scale.
TEST(TrifocalTensor2, tensorRegression) {
  trifocal::TrifocalTestData data = trifocal::getTestData();
  TrifocalTensor2 test_tensor = trifocal::getTestData().gt_tensor;

  // calculate trifocal tensor
  TrifocalTensor2 T = TrifocalTensor2::FromBearingMeasurements(
      data.measurements[0], data.measurements[1], data.measurements[2]);

  const auto expected_tensor = test_tensor.tensor();
  const auto actual_tensor = T.tensor();
  const double scale = expected_tensor.first(0, 0) / actual_tensor.first(0, 0);
  const Matrix2 actual_tensor_scaled0 = scale * actual_tensor.first;
  const Matrix2 actual_tensor_scaled1 = scale * actual_tensor.second;

  EXPECT(assert_equal(expected_tensor.first, actual_tensor_scaled0, 1e-2));
  EXPECT(assert_equal(expected_tensor.second, actual_tensor_scaled1, 1e-2));
}

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
  const std::pair<Matrix2, Matrix2> test_tensor =
      trifocal::getTestData().gt_tensor.tensor();

  // Compute the manifold representation from the tensor.
  const std::pair<Matrix2, Matrix2> actual_tensor =
      TrifocalTensor2::FromTensor(test_tensor.first, test_tensor.second)
          .tensor();

  double scale = test_tensor.first(0, 0) / actual_tensor.first(0, 0);

  const Matrix2 actual_tensor_mat0 = scale * actual_tensor.first;
  const Matrix2 actual_tensor_mat1 = scale * actual_tensor.second;

  // Get the tensor back from the manifold representation, comapre to original.
  EXPECT(assert_equal(test_tensor.first, actual_tensor_mat0));
  EXPECT(assert_equal(test_tensor.second, actual_tensor_mat1));
}

Point2 tranformBearing(const TrifocalTensor2& tensor, const Point2& theta_b,
                     const Point2& theta_c) {
  return tensor.transform(theta_b, theta_c);
}

// Check jacobian of transform()
TEST(TrifocalTensor2, transformJacobian) {
  trifocal::TrifocalTestData data = trifocal::getTestData();
  Rot2 theta_b = data.measurements[0][0], theta_c = data.measurements[1][0];
  Point2 bp(theta_b.c(), theta_b.s()), cp(theta_c.c(), theta_c.s());
  std::function<Point2(const TrifocalTensor2&)> f =
      std::bind(&tranformBearing, std::placeholders::_1, bp, cp);

  Matrix25 expected_H = numericalDerivative11(f, data.gt_tensor);

  Matrix25 actual_H;
  data.gt_tensor.transform(bp, cp, actual_H);
  EXPECT(assert_equal(expected_H, actual_H));
}

// Check jacobian of the full tensor wrt manifold.
TEST(TrifocalTensor2, tensorConversionJacobian) {
  TrifocalTensor2 test_tensor = trifocal::getTestData().gt_tensor;
  std::function<Vector8(const TrifocalTensor2&)> f2 =
      std::bind(trifocal::tensorPairToVector, std::placeholders::_1);
  Matrix85 expected_H_pair = numericalDerivative11(f2, test_tensor);

  Matrix85 actual_H_pair;
  test_tensor.tensor(actual_H_pair);
  EXPECT(assert_equal(expected_H_pair, actual_H_pair, 1e-7));
}

// Check Jacobian of retract.
TEST(TrifocalTensor2, retractJacobian) {
  TrifocalTensor2 test_tensor = trifocal::getTestData().gt_tensor;
  Vector5 retraction;
  retraction << 1.0, 1.0, 1.0, -1.0, 1.0;

  // First matrix
  std::function<TrifocalTensor2(const TrifocalTensor2&, const Vector5&)> f0 =
      std::bind(&TrifocalTensor2::retract, std::placeholders::_1,
                std::placeholders::_2, boost::none, boost::none);
  Matrix55 expected_Dtensor =
      numericalDerivative21(f0, test_tensor, retraction);
  Matrix55 expected_Dretract =
      numericalDerivative22(f0, test_tensor, retraction);

  Matrix55 actual_Dtensor, actual_Dretract;
  test_tensor.retract(retraction, actual_Dtensor, actual_Dretract);
  EXPECT(assert_equal(expected_Dtensor, actual_Dtensor));
  EXPECT(assert_equal(expected_Dretract, actual_Dretract));
}

// Check Jacobian of localCoordinates().
// Also check this.locaCoordinates(this.retract(v)) == v.
TEST(TrifocalTensor2, localCoordinatesJacobian) {
  TrifocalTensor2 test_tensor = trifocal::getTestData().gt_tensor;
  Vector5 v;
  v << 1.0, 1.0, 1.0, -1.0, 1.0;
  TrifocalTensor2 new_tensor = test_tensor.retract(v);

  // First matrix
  std::function<Vector5(const TrifocalTensor2&, const TrifocalTensor2&)> f0 =
      std::bind(&TrifocalTensor2::localCoordinates, std::placeholders::_1,
                std::placeholders::_2, boost::none, boost::none);
  Matrix55 expected_Dtest = numericalDerivative21(f0, test_tensor, new_tensor);
  Matrix55 expected_Dnew = numericalDerivative22(f0, test_tensor, new_tensor);

  Matrix55 actual_Dtest, actual_Dnew;
  Vector5 actual_retraction =
      test_tensor.localCoordinates(new_tensor, actual_Dtest, actual_Dnew);
  EXPECT(assert_equal(v, actual_retraction));
  EXPECT(assert_equal(expected_Dtest, actual_Dtest));
  EXPECT(assert_equal(expected_Dnew, actual_Dnew));
}

int main() {
  TestResult tr;
  return TestRegistry::runAllTests(tr);
}
