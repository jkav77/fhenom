#pragma once

#include <vector>

#include "fhenom/common.h"

namespace fhenom {
class Tensor {
  std::vector<double> data_;
  fhenom::shape_t shape_;

 public:
  Tensor() = default;
  Tensor(std::vector<double> data, fhenom::shape_t shape);

  //////////////////////////////////////////////////////////////////////////////
  // Getters and Setters

  std::vector<double> getData() const { return data_; };

  fhenom::shape_t getShape() const { return shape_; };

  void reshape(shape_t shape);

  void setData(std::vector<double> data, fhenom::shape_t shape);

  double get(shape_t coordinates) const;
};
}  // namespace fhenom
