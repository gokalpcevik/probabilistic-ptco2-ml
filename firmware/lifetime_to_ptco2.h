#pragma once
#include "arm_math_types.h"
#include "arm_nn_types.h"
#include "arm_nnfunctions.h"
#include "icas/nn/fp32_model_weights.h"

// PtcO2 Regressor Input
struct PtcO2_regressor_x
{
  // Layout = [Lifetime, Temperature, Pulses Applied]
  float32_t tensor[3];
};

class PtcO2_regressor_s8_t
{
public:
  PtcO2_regressor_s8_t() = default;

  // Returns the predicted PtcO2 value in [cmHg]
  static float32_t predict_PtcO2(PtcO2_regressor_x* x);
};

class PtcO2_regressor_t
{
public:
  PtcO2_regressor_t() = default;

  // Returns the predicted PtcO2 value in [cmHg]
  static float32_t predict_PtcO2(PtcO2_regressor_x* x);
};
