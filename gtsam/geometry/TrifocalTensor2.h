/* ----------------------------------------------------------------------------
 * GTSAM Copyright 2010, Georgia Tech Research Corporation,
 * Atlanta, Georgia 30332-0415
 * All Rights Reserved
 * Authors: Frank Dellaert, et al. (see THANKS for the full author list)
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

/**
 * @file    TrifocalTensor2.h
 * @brief   A 2x2x2 trifocal tensor in a plane, for 1D cameras.
 * @author  Zhaodong Yang
 * @author  Akshay Krishnan
 */
// \callgraph

#pragma once

#include <gtsam/base/Matrix.h>
#include <gtsam/geometry/Point2.h>
#include <gtsam/geometry/Rot2.h>

namespace gtsam {

/**
 * @brief A trifocal tensor for 1D cameras in a plane. It encodes the
 * relationship between bearing measurements of a point in the plane observed in
 * 3 1D cameras.
 * @addtogroup geometry
 * \nosubgrouping
 */
class TrifocalTensor2 {
 private:
  // 5 dimensional minimal representation.
  Rot2 aRb_, aRc_, atb_, atc_, btc_;

 public:
  enum { dimension = 5 };  // 5-D manifold.

  TrifocalTensor2() {}

  // Construct from minimal representation.
  TrifocalTensor2(const Rot2& aRb, const Rot2& aRc, const Rot2& atb,
                  const Rot2& atc, const Rot2& btc)
      : aRb_(aRb), aRc_(aRc), atb_(atb), atc_(atc), btc_(btc) {}

  //   // Copy constructor
  //   TrifocalTensor2(const TrifocalTensor2& T)
  //       : TrifocalTensor2(T.aRb_, T.aRc_, T.atb_, T.atc_, T.btc_) {}

  /**
   * @brief Construct from the 2x2x2 tensor (a 2x2 matrix pair).
   *
   * @param matrix0 tensor[0, :, :]
   * @param matrix1 tensor[1, :, :]
   * @param Dtensor optional jacobian with respect to tensor
   * @return TrifocalTensor2
   */
  static TrifocalTensor2 FromTensor(
      const Matrix2& matrix0, const Matrix2& matrix1,
      OptionalJacobian<5, 8> Dtensor = boost::none);

  /**
   * @brief Estimates a tensor from at least 7 bearing measurements in 3
   * cameras. Throws a runtime error if the size of inputs are unequal or less
   * than 7.
   *
   * @param bearings_a bearing measurement in camera a.
   * @param bearings_b bearing measurement in camera b.
   * @param bearings_c bearing measurement in camera c.
   * @return Tensor estimated from the measurements.
   */
  static TrifocalTensor2 FromBearingMeasurements(
      const std::vector<Rot2>& bearings_a, const std::vector<Rot2>& bearings_b,
      const std::vector<Rot2>& bearings_c);

  /**
   * @brief Estimates a tensor from 8 projective measurements in 3 cameras.
   * Throws a runtime error if the size of inputs are unequal or less than 8.
   *
   * @param a projective 1D bearing measurement in camera a.
   * @param b projective 1D bearing measurement in camera b.
   * @param c projective 1D bearing measurement in camera c.
   * @return tensor estimated from the measurements.
   */
  static TrifocalTensor2 FromProjectiveBearingMeasurements(
      const std::vector<Point2>& a, const std::vector<Point2>& b,
      const std::vector<Point2>& c);

  /**
   * @brief Computes the bearing in camera 'a' given bearing measurements in
   * cameras 'b' and 'c'.
   *
   * @param bZp bearing measurement in camera b
   * @param cZp bearing measurement in camera c
   * @return bearing measurement in camera a
   */
  Rot2 transform(const Rot2& bZp, const Rot2& cZp,
                 OptionalJacobian<1, 5> Dtensor = boost::none) const;

  /**
   * @brief Computes the bearing in camera 'a' from that of cameras 'b' and 'c',
   * in projective coordinates.
   *
   * @param bZp projective bearing measurement in camera b
   * @param cZp projective bearing measurement in camera c
   * @return projective bearing measurement in camera a
   */
  Point2 transform(const Point2& bp, const Point2& cp,
                   OptionalJacobian<2, 5> Dtensor = boost::none) const;

  /**
   * @brief Returns the 2x2x2 tensor representation.
   * 
   * @param Dtensor jacobian of output with respect to manifold element.
   */
  std::pair<Matrix2, Matrix2> tensor(
      OptionalJacobian<8, 5> Dtensor = boost::none) const;

  // Map (this tensor + v) from tangent space to the manifold. v is an increment
  // in tangent space.
  TrifocalTensor2 retract(const Vector5& v,
                          OptionalJacobian<5, 5> Dtensor = boost::none,
                          OptionalJacobian<5, 5> Dv = boost::none) const;

  // Difference between another tensor and this tensor in tangent space.
  Vector5 localCoordinates(const TrifocalTensor2& other,
                           OptionalJacobian<5, 5> Dtensor = boost::none,
                           OptionalJacobian<5, 5> Dother = boost::none) const;

  /**
   * @brief the 5 dimensional minimal representation.
   * 
   * @param s: optional starting string
   */
  void print(const std::string& s = "") const;

  // Check whether this tensor equals to another.
  bool equals(const TrifocalTensor2& other, double tol = 1e-9) const;

  // Accessors
  Rot2 aRb() const { return aRb_; }
  Rot2 aRc() const { return aRc_; }
  Rot2 atb() const { return atb_; }
  Rot2 atc() const { return atc_; }
  Rot2 btc() const { return btc_; }
};

template <>
struct traits<TrifocalTensor2> : public internal::Manifold<TrifocalTensor2> {};

template <>
struct traits<const TrifocalTensor2>
    : public internal::Manifold<TrifocalTensor2> {};

}  // namespace gtsam
