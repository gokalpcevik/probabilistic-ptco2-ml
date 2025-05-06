/*
 *      Author: Gokalp Cevik (gevik@wpi.edu)
 */

#ifndef INC_ICAS_MONO_EXP_DECAY_H_
#define INC_ICAS_MONO_EXP_DECAY_H_

#include "icas/matrix.h"
#include "icas/measurement-settings.h"
#include "icas/solver/solver.h"
#include <cstddef>

static constexpr float32_t O2_TIME_SCALE = 50.0f;
static constexpr float32_t O2_LUMINESCENCE_SCALE = 700.0f;

// Mono-exponential data
struct mono_exp_decay_data
{
  float32_t time[NUM_INT];
  float32_t luminescence[NUM_INT];
};

// These are still needed since sometimes we need to calculate the gradient, loss, etc. individually
void mono_exp_gradient(dmatrix_t const& params, dmatrix_t& gradient, void* fit_data);
void mono_exp_jacobian(dmatrix_t const& params, dmatrix_t& J, void* fit_data);
void mono_exp_residuals(dmatrix_t const& params, dmatrix_t& r, void* fit_data);
float32_t mono_exp_loss(dmatrix_t const& params, void* fit_data);

// Compute all iteration terms in one-go (reduce exponential calls)
template <uint8_t TERM_MASK>
void mono_exp_iteration_terms_T(opt_iteration_terms& terms, dmatrix_t const& params, void* fit_data)
{
  mono_exp_decay_data* data = static_cast<mono_exp_decay_data*>(fit_data);
  // Loss is needed by every algorithm
  terms.loss = 0.0f;

  if constexpr (COMPUTE_GRADIENT_MASK & TERM_MASK)
  {
    terms.gradient.set_zero();
  }
  if constexpr (COMPUTE_HESSIAN_MASK & TERM_MASK)
  {
    terms.H.set_zero();
  }

  for (size_t i = 0; i < NUM_INT; ++i)
  {
    float32_t exp_term = params.at(1) * expf(params.at(2) * data->time[i]); // Exponential term
    float32_t prediction =
        params.at(0) + exp_term; // Prediction = DC Term + Scaling * exp (Rate * time), lifetime = -1.0/rate
    float32_t residual = prediction - data->luminescence[i]; // Residual = Prediction - observation
    float32_t drv_r_wrt_beta2 =
        params.at(1) * data->time[i] * exp_term; // Derivative of the residual w.r.t. to the third term (exp. rate)
    // Jacobian
    if constexpr (COMPUTE_JACOBIAN_MASK & TERM_MASK)
    {
      terms.J.at(i, 0) = 1.0f;
      terms.J.at(i, 1) = exp_term;
      terms.J.at(i, 2) = drv_r_wrt_beta2;
    }
    // Vector of residuals
    if constexpr (COMPUTE_RESIDUAL_MASK & TERM_MASK)
    {
      terms.r.at(i) = residual;
    }
    // Gauss-Newton approximation to the Hessian
    if constexpr (COMPUTE_HESSIAN_MASK & TERM_MASK)
    {
      // Accumulate outer product of each row of J to form the Hessian approximation
      // H[m,n] += J[i,m] * J[i,n] for this data point
      terms.H.at(0, 0) += 1.0f;            // 1 * 1
      terms.H.at(0, 1) += exp_term;        // 1 * exp_term
      terms.H.at(0, 2) += drv_r_wrt_beta2; // 1 * dr/dbeta2

      terms.H.at(1, 0) += exp_term;                   // exp_term * 1
      terms.H.at(1, 1) += exp_term * exp_term;        // exp_term * exp_term
      terms.H.at(1, 2) += exp_term * drv_r_wrt_beta2; // exp_term * I_i

      terms.H.at(2, 0) += drv_r_wrt_beta2;                   // I_i * 1
      terms.H.at(2, 1) += drv_r_wrt_beta2 * exp_term;        // I_i * exp_term
      terms.H.at(2, 2) += drv_r_wrt_beta2 * drv_r_wrt_beta2; // I_i * I_i
    }
    // Gradient
    if constexpr (COMPUTE_GRADIENT_MASK & TERM_MASK)
    {
      terms.gradient.at(0) += residual;
      terms.gradient.at(1) += residual * exp_term;
      terms.gradient.at(2) += residual * drv_r_wrt_beta2;
    }
    if constexpr (COMPUTE_LOSS_MASK & TERM_MASK)
    {
      terms.loss += residual * residual;
    }
  }
  if constexpr (COMPUTE_LOSS_MASK & TERM_MASK)
  {
    terms.loss *= 0.5f;
  }
}

// This is a bit hard to read, but it is more optimized than the above version
template <uint8_t TERM_MASK>
void mono_exp_iteration_terms_optimized_T(opt_iteration_terms& terms, dmatrix_t const& params, void* fit_data)
{
  mono_exp_decay_data* data = static_cast<mono_exp_decay_data*>(fit_data);

  // Cache frequently accessed parameters in registers
  const float32_t beta0 = params.at(0); // DC term
  const float32_t beta1 = params.at(1); // Scaling
  const float32_t beta2 = params.at(2); // Rate

  // Single-pass accumulation using registers for intermediate values
  float32_t loss_acc = 0.0f;
  float32_t grad0_acc = 0.0f;
  float32_t grad1_acc = 0.0f;
  float32_t grad2_acc = 0.0f;

  // Hessian accumulators - upper triangular only due to symmetry
  float32_t H01 = 0.0f; // H(0,1) = H(1,0)
  float32_t H02 = 0.0f; // H(0,2) = H(2,0)
  float32_t H11 = 0.0f; // H(1,1)
  float32_t H12 = 0.0f; // H(1,2) = H(2,1)
  float32_t H22 = 0.0f; // H(2,2)

  // Get direct pointers to avoid repeated address calculations
  const float32_t* time_ptr = data->time;
  const float32_t* lum_ptr = data->luminescence;

  // Process all points in a single pass, keeping values in registers
  for (size_t i = 0; i < NUM_INT; ++i)
  {
    // Load input values once
    const float32_t time_val = time_ptr[i];
    const float32_t lum_val = lum_ptr[i];

    // Compute exponential term - this is used multiple times
    const float32_t exp_arg = beta2 * time_val;
    const float32_t exp_term = beta1 * expf(exp_arg);

    // Compute prediction and residual
    const float32_t pred = beta0 + exp_term;
    const float32_t residual = pred - lum_val;

    // Compute derivative term - used in both Jacobian and gradient
    const float32_t drv_r_beta2 = exp_term * time_val;

    // Update Jacobian if needed
    if constexpr (COMPUTE_JACOBIAN_MASK & TERM_MASK)
    {
      terms.J.at(i, 0) = 1.0f;
      terms.J.at(i, 1) = exp_term;
      terms.J.at(i, 2) = drv_r_beta2;
    }

    // Store residual if needed
    if constexpr (COMPUTE_RESIDUAL_MASK & TERM_MASK)
    {
      terms.r.at(i) = residual;
    }

    // Accumulate Hessian terms
    if constexpr (COMPUTE_HESSIAN_MASK & TERM_MASK)
    {
      // Exploit symmetry by only computing upper triangle
      H01 += exp_term;
      H02 += drv_r_beta2;
      H11 += exp_term * exp_term;
      H12 += exp_term * drv_r_beta2;
      H22 += drv_r_beta2 * drv_r_beta2;
    }

    // Accumulate gradient components
    if constexpr (COMPUTE_GRADIENT_MASK & TERM_MASK)
    {
      grad0_acc += residual;
      grad1_acc += residual * exp_term;
      grad2_acc += residual * drv_r_beta2;
    }

    // Accumulate loss
    if constexpr (COMPUTE_LOSS_MASK & TERM_MASK)
    {
      loss_acc += residual * residual;
    }
  }

  // Final updates to output structures
  if constexpr (COMPUTE_LOSS_MASK & TERM_MASK)
  {
    terms.loss = 0.5f * loss_acc;
  }

  if constexpr (COMPUTE_GRADIENT_MASK & TERM_MASK)
  {
    terms.gradient.at(0) = grad0_acc;
    terms.gradient.at(1) = grad1_acc;
    terms.gradient.at(2) = grad2_acc;
  }

  if constexpr (COMPUTE_HESSIAN_MASK & TERM_MASK)
  {
    // Fill symmetric Hessian matrix using accumulated values
    terms.H.at(0, 0) = NUM_INT; // Constant term
    terms.H.at(0, 1) = terms.H.at(1, 0) = H01;
    terms.H.at(0, 2) = terms.H.at(2, 0) = H02;
    terms.H.at(1, 1) = H11;
    terms.H.at(1, 2) = terms.H.at(2, 1) = H12;
    terms.H.at(2, 2) = H22;
  }
}

#endif
