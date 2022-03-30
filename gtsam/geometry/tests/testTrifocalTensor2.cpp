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
#include <gtsam/nonlinear/Expression.h>
#include <gtsam/nonlinear/ExpressionFactor.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>

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
  data.gt_poses.emplace_back(0.5, 2, 2 * acos(0.0) / 6);
  data.gt_poses.emplace_back(2.1, 5, 2 * acos(0.0) / 3);

  Rot2 aRb = data.gt_poses[1].rotation();
  Rot2 aRc = data.gt_poses[2].rotation();
  Rot2 atb = bearingFromTranslation(data.gt_poses[1].translation());
  Rot2 atc = bearingFromTranslation(data.gt_poses[2].translation());
  Rot2 btc = bearingFromTranslation(
      data.gt_poses[1].transformTo(data.gt_poses[2].translation()));
  data.gt_tensor = TrifocalTensor2(aRb, aRc, atb, atc, btc);

  // Landmarks
  data.gt_landmarks.emplace_back(2.0, 8.1);
  data.gt_landmarks.emplace_back(2.4, 15.0);
  data.gt_landmarks.emplace_back(10.6, 22.2);
  data.gt_landmarks.emplace_back(9.2, 35.5);
  data.gt_landmarks.emplace_back(7.1, 24.6);
  data.gt_landmarks.emplace_back(15.5, 30.8);
  data.gt_landmarks.emplace_back(8.1, 19.1);

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

// Testing equals() method.
TEST(TrifocalTensor2, equals) {
  TrifocalTensor2 test_tensor = trifocal::getTestData().gt_tensor;
  TrifocalTensor2 same_tensor = TrifocalTensor2(test_tensor);

  EXPECT(test_tensor.equals(same_tensor));  // same tensors are equal.
  EXPECT(!test_tensor.equals(
      TrifocalTensor2()));  // different tensors are unequal.
}

// Check correct minimal representation obtained when estimating linearly.
TEST(TrifocalTensor2, linearEstimationMinimalRepresentation) {
  trifocal::TrifocalTestData data = trifocal::getTestData();

  // Compute trifocal tensor
  TrifocalTensor2 T = TrifocalTensor2::FromBearingMeasurements(
      data.measurements[0], data.measurements[1], data.measurements[2]);

  for (size_t i = 0; i < data.measurements[0].size(); i++) {
    std::cout << data.measurements[0][i].theta() << " "
              << data.measurements[1][i].theta() << " "
              << data.measurements[2][i].theta() << std::endl;
  }
  data.gt_poses[1].print("view1 pose:");
  T.print("actual tensor");
  data.gt_tensor.print("GT tensor");
  EXPECT(T.equals(data.gt_tensor));
}

// Check the correct tensor is computed from measurements (catch regressions).
// The tensor is computed only upto a scale.
// This should pass if LinearEstimationMinimalRepresentation passes.
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

// Check transform() correctly transforms bearing measurements from views B, C
// to A.
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

// Check transform() correctly transforms measurements in projective 2D space
// from views B, C to A.
TEST(TrifocalTensor2, transform_Point2) {
  trifocal::TrifocalTestData data = trifocal::getTestData();

  // Compute trifocal tensor
  TrifocalTensor2 T = TrifocalTensor2::FromBearingMeasurements(
      data.measurements[0], data.measurements[1], data.measurements[2]);

  // Estimate measurement in view A from measurements in B and C
  for (unsigned int i = 0; i < data.measurements[0].size(); i++) {
    const Point2 actual_measurement = T.transform(
        data.measurements[1][i].unit(), data.measurements[2][i].unit());

    // TODO(zhaodong): how do we fix this?
    // there might be two solutions for u1 and u2, comparing the ratio instead
    // of both cos and sin
    EXPECT(assert_equal(actual_measurement(0) * data.measurements[0][i].s(),
                        actual_measurement(1) * data.measurements[0][i].c(),
                        1e-8));
  }
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

// Check jacobian of transform() for projective measurements.
TEST(TrifocalTensor2, transformProjectiveJacobian) {
  trifocal::TrifocalTestData data = trifocal::getTestData();
  Rot2 theta_b = data.measurements[0][0], theta_c = data.measurements[1][0];
  Point2 bp(theta_b.c(), theta_b.s()), cp(theta_c.c(), theta_c.s());
  std::function<Point2(const TrifocalTensor2&)> f =
      std::bind(&tranformBearing, std::placeholders::_1, bp, cp);

  Matrix25 expected_H = numericalDerivative11(f, data.gt_tensor);

  Matrix25 actual_H;
  data.gt_tensor.transform(bp, cp, actual_H);
  EXPECT(assert_equal(expected_H, actual_H, 1e-6));
}

// Check jacobian of transform() for bearing measurements.
TEST(TrifocalTensor2, transformBearingJacobian) {
  trifocal::TrifocalTestData data = trifocal::getTestData();
  Rot2 theta_b = data.measurements[0][0], theta_c = data.measurements[1][0];
  std::function<Rot2(const TrifocalTensor2&)> f =
      [&](const TrifocalTensor2& T) {
        return T.transform(theta_b, theta_c, boost::none);
      };

  Matrix15 expected_H = numericalDerivative11(f, data.gt_tensor);

  Matrix15 actual_H;
  data.gt_tensor.transform(theta_b, theta_c, actual_H);
  EXPECT(assert_equal(expected_H, actual_H, 1e-7));
}

// Check jacobian of the full tensor wrt manifold.
TEST(TrifocalTensor2, tensorConversionJacobian) {
  TrifocalTensor2 test_tensor = trifocal::getTestData().gt_tensor;
  std::function<Vector8(const TrifocalTensor2&)> f2 =
      std::bind(trifocal::tensorPairToVector, std::placeholders::_1);
  Matrix85 expected_H_pair = numericalDerivative11(f2, test_tensor);

  Matrix85 actual_H_pair;
  test_tensor.tensor(actual_H_pair);
  EXPECT(assert_equal(expected_H_pair, actual_H_pair, 1e-6));
}

// Check this.locaCoordinates(this.retract(v)) == v.
TEST(TrifocalTensor2, localCoordinatesOfRetract) {
  TrifocalTensor2 test_tensor = trifocal::getTestData().gt_tensor;
  Vector5 expected_v;
  expected_v << 1.0, 1.0, 1.0, -1.0, 1.0;
  Vector5 actual_v =
      test_tensor.localCoordinates(test_tensor.retract(expected_v));
  EXPECT(assert_equal(expected_v, actual_v));
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
  test_tensor.localCoordinates(new_tensor, actual_Dtest, actual_Dnew);
  EXPECT(assert_equal(expected_Dtest, actual_Dtest));
  EXPECT(assert_equal(expected_Dnew, actual_Dnew));
}

TEST(TrifocalTensor2, optimizationExpressionFactor) {
  const trifocal::TrifocalTestData data = trifocal::getTestData();

  // calculate trifocal tensor
  TrifocalTensor2 T_init = TrifocalTensor2::FromBearingMeasurements(
      data.measurements[0], data.measurements[1], data.measurements[2]);
  Expression<TrifocalTensor2> T_(1);

  NonlinearFactorGraph graph;
  Values initial;
  initial.insert(1, T_init);
  SharedNoiseModel noise_model = noiseModel::Isotropic::Sigma(1, 0.01);

  for (int i = 0; i < data.gt_landmarks.size(); i++) {
    auto transform_fn = [&data, i](const TrifocalTensor2& T,
                                   OptionalJacobian<1, 5> H) {
      return T.transform(data.measurements[1][i], data.measurements[2][i], H);
    };
    Expression<Rot2> u_(transform_fn, T_);
    // std::function<double(const &, OptionalJacobian<1, 2>)> dot_product =
    //     [&data, i](const Point2& p1, OptionalJacobian<1, 2> H1) {
    //       const Point2 p2 = projective(data.measurements[0][i]);
    //       if (H1) {
    //         (*H1)(0) = p2.x();
    //         (*H1)(1) = p2.y();
    //       }
    //       return p1.dot(p2);
    //     };
    // Expression<double> error_(dot_product, u_);
    ExpressionFactor<Rot2> f(noise_model, data.measurements[0][i], u_);
    graph.push_back(f);
  }
  LevenbergMarquardtOptimizer optimizer(graph, initial);
  Values result = optimizer.optimize();
  TrifocalTensor2 resultT = result.at<TrifocalTensor2>(1);
  resultT.print("result");
  T_init.print("initial");
  data.gt_tensor.print("GT");
  EXPECT(T_init.equals(resultT));
  EXPECT(data.gt_tensor.equals(resultT));
}

int main() {
  TestResult tr;
  return TestRegistry::runAllTests(tr);
}
