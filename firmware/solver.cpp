/*
 *      Author: Gokalp Cevik (gevik@wpi.edu)
 */

#include "icas/solver/solver.h"

char const* opt_termination_reason_to_string(opt_termination_reason reason)
{
  switch (reason)
  {
  case OPT_CONVERGED:
    return "CONVERGED";
  case OPT_ILL_CONDITIONED:
    return "ILL_CONDITIONED";
  case OPT_MAX_ITERATIONS_REACHED:
    return "MAX_ITER_REACHED";
  }
  return "OPT_INVALID_REASON";
}

char const* opt_algorithm_to_string(opt_algorithm algorithm)
{
  switch (algorithm)
  {
  case OPT_LEVENBERG_MARQUARDT:
    return "Levenberg-Marquardt";
  case OPT_GAUSS_NEWTON:
    return "Gauss-Newton";
  case OPT_GRADIENT_DESCENT:
    return "Gradient Descent";
  case OPT_GD_MOMENTUM:
    return "Momemtum GD";
  case OPT_GD_ADAM:
    return "Adam";
  case OPT_BFGS:
    return "BFGS";
  default:
    return "[not supported]";
  }
}

void clip_gradient_norm(dmatrix_t& gradient, float32_t max_norm)
{
  float32_t grad_norm = gradient.norm();
  if (grad_norm > max_norm)
  {
    gradient *= (max_norm / grad_norm);
  }
}

// Armijo line-search
float32_t armijo_line_search(opt_context const* ctx, dmatrix_t const& params, dmatrix_t const& search_direction,
                             dmatrix_t const& gradient, float32_t f_initial, int max_iter = 10,
                             float32_t initial_alpha = 1.0f, float32_t c = 0.0001f, float32_t rho = 0.7f)
{
  float32_t alpha = initial_alpha;
  // Compute directional derivative g^T * p
  float32_t dir_derivative = gradient.dot(search_direction);
  // Start backtracking
  dmatrix_t new_params = params + alpha * search_direction;
  float32_t f_new = ctx->loss(new_params, ctx->fit_data);
  for (int i = 0; i < max_iter; ++i)
  {
    if (f_new <= f_initial + c * alpha * dir_derivative)
    {
      break;
    }
    alpha *= rho; // Reduce step size
    new_params = params + alpha * search_direction;
    f_new = ctx->loss(new_params, ctx->fit_data);
  }
  return alpha;
}

// Wolfe's condition line-search: Armijo + Curvature condition
float32_t wolfe_line_search(opt_context const* ctx, dmatrix_t const& params, dmatrix_t const& search_direction,
                            dmatrix_t const& gradient, float32_t f_initial, int max_iter = 10,
                            float32_t initial_alpha = 1.0f, float32_t c1 = 0.0001f, float32_t c2 = 0.9f,
                            float32_t rho = 0.5f)
{
  float32_t alpha = initial_alpha;
  // Compute directional derivative g^T * p
  float32_t dir_derivative = gradient.dot(search_direction);
  // Start backtracking
  dmatrix_t new_params = params + alpha * search_direction;
  float32_t f_new = ctx->loss(new_params, ctx->fit_data);
  dmatrix_t new_grad = dmatrix_t(ctx->n_params, 1);
  ctx->gradient(new_params, new_grad, ctx->fit_data);
  for (int i = 0; i < max_iter; ++i)
  {
    // First term is Armijo's condition, second term is the curvature condition
    if (f_new <= f_initial + c1 * alpha * dir_derivative && new_grad.dot(search_direction) >= c2 * dir_derivative)
    {
      break;
    }
    alpha *= rho; // Reduce step size
    // Recalculate Parameters
    new_params = params + alpha * search_direction;
    f_new = ctx->loss(new_params, ctx->fit_data);
    ctx->gradient(new_params, new_grad, ctx->fit_data);
  }
  return alpha;
}

#if SOLVER_ENABLE_LMA==1
float32_t lambda0(const dmatrix_t& H_approx, float32_t mean_diag_scaling_factor = 1e-3f)
{
  // Use diagonal elements of H_approx (J^T * J) to get problem scale
  // We already compute this anyway in the main loop
  float32_t mean_diagonal = H_approx.diagonal().mean();
  // Start with a small fraction of the mean curvature
  // Using 1e-3 as starting factor tends to work well in practice
  return mean_diagonal * mean_diag_scaling_factor;
}

float32_t lma_update_lambda(float32_t lambda, float32_t rho, float32_t rho_good_th, float32_t rho_bad_th,
                            float32_t lambda_increase_factor, float32_t lambda_decrease_factor)
{
  // NOLINTBEGIN
  if (rho > rho_good_th)
  {
    return std::max(lambda * lambda_decrease_factor, 1e-7f);
  }
  else if (rho < rho_bad_th)
  {
    return std::min(lambda * lambda_increase_factor, 1e+7f);
  }
  // NOLINTEND
  return lambda;
}

void opt_levenberg_marquardt_solve(const opt_config* cfg, opt_context* ctx, opt_state* state)
{
  opt_lma_config* lma_cfg = static_cast<opt_lma_config*>(cfg->algorithm_config);
  dmatrix_t const I_n = dmatrix_t::identity(ctx->n_params);
  opt_iteration_terms terms(ctx->n_params, ctx->n_data, LEVENBERG_MARQUARDT_TERMS_MASK);
  ctx->iteration_terms(terms, ctx->params, ctx->fit_data);
  state->loss = terms.loss;
  float32_t prev_loss = state->loss; // Track previous iteration's loss
  float32_t lambda = lambda0(terms.H, lma_cfg->mean_diagonal_scaling_factor);
  bool params_changed = false;
  for (state->iterations = 0; state->iterations < cfg->max_iterations; ++state->iterations)
  {
    if (state->iterations > 0 && params_changed)
    {
      ctx->iteration_terms(terms, ctx->params, ctx->fit_data);
      state->loss = terms.loss;
      // Check relative improvement from previous iteration
      float32_t rel_improvement = (prev_loss - state->loss) / prev_loss;
      if (rel_improvement < 1e-3f && state->iterations > 4) // Allow a few iterations to get started
      {
        state->termination_reason = OPT_CONVERGED;
        return;
      }
      prev_loss = state->loss;
    }
    dmatrix_t N = (terms.H) + (I_n * lambda);
    if (N.diagonal().min_coeff() < cfg->min_diagonal)
    {
      state->termination_reason = OPT_ILL_CONDITIONED;
      return;
    }
    dmatrix_t N_inv = dmatrix_t::zero(ctx->n_params, ctx->n_params);
    if (!N.inverse(N_inv))
    {
      state->termination_reason = OPT_ILL_CONDITIONED;
      return;
    }
    dmatrix_t delta = -1.0f * (N_inv * terms.gradient);
    float32_t term1 = -1.0f * terms.gradient.dot(delta);
    float32_t term2 = -0.5f * delta.dot(terms.H * delta);
    float32_t pred_reduction = term1 + term2;
    if (fabsf(pred_reduction) < 1e-4f * state->loss) // Less strict threshold
    {
      lambda = std::min(lambda * lma_cfg->lambda_increase_factor, 1e+7f);
      params_changed = false;
      continue;
    }
    float32_t update_step_size = delta.norm();
    if (update_step_size < cfg->min_update_step)
    {
      state->termination_reason = OPT_CONVERGED;
      return;
    }
    float32_t new_err = ctx->loss(ctx->params + delta, ctx->fit_data);
    float32_t actual_reduction = state->loss - new_err;
    if (new_err < state->loss)
    {
      ctx->params += delta;
      state->loss = new_err;
      lambda = lma_update_lambda(lambda, pred_reduction / actual_reduction, lma_cfg->rho_good_th, lma_cfg->rho_bad_th,
                                 lma_cfg->lambda_increase_factor, lma_cfg->lambda_decrease_factor);
      params_changed = true;
    }
    else
    {
      lambda = std::min(lambda * lma_cfg->lambda_increase_factor, 1e+7f);
      params_changed = false;
    }
  }
  state->termination_reason = OPT_MAX_ITERATIONS_REACHED;
}
#endif

#if SOLVER_ENABLE_GNA==1
void opt_gauss_newton_solve(const opt_config* cfg, opt_context* ctx, opt_state* state)
{
  opt_gna_config* gna_cfg = static_cast<opt_gna_config*>(cfg->algorithm_config);
  // Pre-compute the damped identity matrix once, since damping is constant
  const dmatrix_t damped_I = dmatrix_t::identity(ctx->n_params) * gna_cfg->damping;
  // Initialize terms outside the loop with only the masks we need
  opt_iteration_terms terms(ctx->n_params, ctx->n_data, GAUSS_NEWTON_TERMS_MASK);
  // Track previous loss to check convergence
  float32_t prev_loss = INFINITY;
  for (state->iterations = 0; state->iterations < cfg->max_iterations; ++state->iterations)
  {
    // Compute terms for current iteration
    ctx->iteration_terms(terms, ctx->params, ctx->fit_data);
    state->loss = terms.loss;
    // Check for relative improvement after a few iterations
    if (state->iterations > 3 && ((prev_loss - state->loss) / prev_loss < 1e-3f))
    {
      state->termination_reason = OPT_CONVERGED;
      return;
    }
    prev_loss = state->loss;
    // Add damping to Hessian - using pre-computed damped identity
    dmatrix_t N = terms.H + damped_I;
    // Combined ill-conditioning checks
    if (N.diagonal().min_coeff() < cfg->min_diagonal || !N.inverse(N))
    {
      state->termination_reason = OPT_ILL_CONDITIONED;
      return;
    }
    // Compute the update step
    dmatrix_t delta = -1.0f * (N * terms.gradient);
    // Perform line search and scale delta
    // float32_t alpha = armijo_line_search(ctx, ctx->params, delta, terms.gradient, state->loss);
    float alpha = 0.008f;
    delta *= alpha;
    // Check convergence based on step size
    if (delta.norm() < cfg->min_update_step)
    {
      state->termination_reason = OPT_CONVERGED;
      return;
    }
    // Update parameters
    ctx->params += delta;
  }
  state->loss = ctx->loss(ctx->params, ctx->fit_data);
  state->termination_reason = OPT_MAX_ITERATIONS_REACHED;
}
#endif

#if SOLVER_ENABLE_GD==1
void opt_gradient_descent_solve(const opt_config* cfg, opt_context* ctx, opt_state* state)
{
  // Initialize terms with only the masks needed for gradient descent
  opt_iteration_terms terms(ctx->n_params, ctx->n_data, GD_TERMS_MASK);
  // Track previous loss for convergence checking
  float32_t prev_loss = INFINITY;
  for (state->iterations = 0; state->iterations < cfg->max_iterations; ++state->iterations)
  {
    // Compute terms for current iteration
    ctx->iteration_terms(terms, ctx->params, ctx->fit_data);
    state->loss = terms.loss;
    // Check for relative improvement after initial iterations
    if (state->iterations > 50 && ((prev_loss - state->loss) / prev_loss < 1e-3f))
    {
      state->termination_reason = OPT_CONVERGED;
      return;
    }
    prev_loss = state->loss;
    // Clip gradient norm to prevent numerical instability
    clip_gradient_norm(terms.gradient, 1.0f);
    // Compute descent direction and perform line search in one step
    // Note: We can combine these operations since gradient descent's direction
    // is simply the negative gradient
    float32_t alpha = wolfe_line_search(ctx, ctx->params,
                                        -1.0f * terms.gradient, // Direction is negative gradient
                                        terms.gradient, state->loss);
    // float alpha = 0.008f;
    //   Compute the actual update step
    dmatrix_t delta = terms.gradient * (-alpha); // Combine scaling into one operation
    // Check convergence based on step size
    float32_t step_size = delta.norm();
    if (step_size < cfg->min_update_step)
    {
      state->termination_reason = OPT_CONVERGED;
      return;
    }
    // Update parameters
    ctx->params += delta;
  }
  state->loss = ctx->loss(ctx->params, ctx->fit_data);
  state->termination_reason = OPT_MAX_ITERATIONS_REACHED;
}
#endif

#if SOLVER_ENABLE_GD1O==1
void opt_gd_momentum_solve(const opt_config* cfg, opt_context* ctx, opt_state* state)
{
  opt_gd1o_config* gd1o_cfg = static_cast<opt_gd1o_config*>(cfg->algorithm_config);

  // Initialize momentum vector and terms structure with appropriate masks
  dmatrix_t m = dmatrix_t::zero(ctx->n_params, 1);
  opt_iteration_terms terms(ctx->n_params, ctx->n_data, GD_MOMENTUM_TERMS_MASK);

  // Pre-compute the momentum decay factor for efficiency
  const float32_t momentum_decay = 1.0f - gd1o_cfg->mu;

  float32_t prev_loss = INFINITY;

  for (state->iterations = 0; state->iterations < cfg->max_iterations; ++state->iterations)
  {
    // Compute terms for current iteration
    ctx->iteration_terms(terms, ctx->params, ctx->fit_data);
    state->loss = terms.loss;

    // Check for relative improvement after initial iterations
    if (state->iterations > 50 && ((prev_loss - state->loss) / prev_loss < 1e-3f))
    {
      state->termination_reason = OPT_CONVERGED;
      return;
    }
    prev_loss = state->loss;

    // Clip gradients early, just like in gradient descent
    clip_gradient_norm(terms.gradient, 1.0f);

    // Update momentum using pre-computed decay factor
    m = (gd1o_cfg->mu * m) + (momentum_decay * terms.gradient);

    // Compute and apply bias correction
    float32_t iter_plus_one = (float32_t)(state->iterations + 1);
    float32_t mu_correction = 1.0f / (1.0f - powf(gd1o_cfg->mu, iter_plus_one));
    dmatrix_t m_corrected = m * mu_correction;

    // Perform line search using the momentum-adjusted gradient
    float32_t alpha = wolfe_line_search(ctx, ctx->params,
                                        -1.0f * m_corrected, // Direction is negative momentum gradient
                                        m_corrected,         // Use momentum gradient for line search
                                        state->loss);
    // float alpha = 0.008f;

    // Compute the actual update step
    dmatrix_t delta = m_corrected * (-alpha);

    // Check convergence based on step size
    float32_t step_size = delta.norm();
    if (step_size < cfg->min_update_step)
    {
      state->termination_reason = OPT_CONVERGED;
      return;
    }

    // Update parameters
    ctx->params += delta;
  }

  state->loss = ctx->loss(ctx->params, ctx->fit_data);
  state->termination_reason = OPT_MAX_ITERATIONS_REACHED;
}
#endif

#if SOLVER_ENABLE_ADAM==1
void opt_gd_adam_solve(const opt_config* cfg, opt_context* ctx, opt_state* state)
{
  opt_gd_adam_config* gd_adam_cfg = static_cast<opt_gd_adam_config*>(cfg->algorithm_config);

  // Initialize moment estimates and terms structure
  dmatrix_t m = dmatrix_t::zero(ctx->n_params, 1); // First moment (mean)
  dmatrix_t v = dmatrix_t::zero(ctx->n_params, 1); // Second moment (variance)
  opt_iteration_terms terms(ctx->n_params, ctx->n_data, GD_ADAM_TERMS_MASK);

  // Pre-compute the decay factors for efficiency
  const float32_t mu_decay = 1.0f - gd_adam_cfg->mu;
  const float32_t nu_decay = 1.0f - gd_adam_cfg->nu;
  const float32_t eps = 1e-7f;

  float32_t prev_loss = INFINITY;

  for (state->iterations = 0; state->iterations < cfg->max_iterations; ++state->iterations)
  {
    // Compute terms for current iteration
    ctx->iteration_terms(terms, ctx->params, ctx->fit_data);
    state->loss = terms.loss;

    // Check for relative improvement after initial iterations
    // Changed to 50 iterations to match gradient descent implementation
    if (state->iterations > 50 && ((prev_loss - state->loss) / prev_loss < 1e-3f))
    {
      state->termination_reason = OPT_CONVERGED;
      return;
    }
    prev_loss = state->loss;

    // Clip gradients early, just like in gradient descent
    clip_gradient_norm(terms.gradient, 1.0f);

    // Update first moment estimate (mean)
    m = (gd_adam_cfg->mu * m) + (mu_decay * terms.gradient);

    // Update second moment estimate (variance)
    dmatrix_t grad_squared = terms.gradient.cwiseProduct(terms.gradient);
    v = (gd_adam_cfg->nu * v) + (nu_decay * grad_squared);

    // Compute bias correction factors
    float32_t iter_plus_one = (float32_t)(state->iterations + 1);
    float32_t mu_correction = 1.0f / (1.0f - powf(gd_adam_cfg->mu, iter_plus_one));
    float32_t nu_correction = 1.0f / (1.0f - powf(gd_adam_cfg->nu, iter_plus_one));

    // Compute Adam update direction
    dmatrix_t m_corrected = m * mu_correction;
    dmatrix_t v_corrected = v * nu_correction;
    dmatrix_t v_sqrt = v_corrected.cwiseSqrt();
    v_sqrt += dmatrix_t::constant(ctx->n_params, 1, eps);

    // Compute the descent direction using Adam's formula
    dmatrix_t adam_grad = m_corrected.cwiseQuotient(v_sqrt);

    // Perform line search using the Adam-adjusted gradient
    float32_t alpha = wolfe_line_search(ctx, ctx->params,
                                        -1.0f * adam_grad, // Direction is negative Adam gradient
                                        adam_grad,         // Use Adam gradient for line search
                                        state->loss);
    // float alpha = 0.008f;

    // Compute the actual update step
    dmatrix_t delta = adam_grad * (-alpha);

    // Check convergence based on step size
    if (delta.norm() < cfg->min_update_step)
    {
      state->termination_reason = OPT_CONVERGED;
      return;
    }

    // Update parameters
    ctx->params += delta;
  }

  state->loss = ctx->loss(ctx->params, ctx->fit_data);
  state->termination_reason = OPT_MAX_ITERATIONS_REACHED;
}
#endif

#if SOLVER_ENABLE_BFGS==1
void opt_bfgs_solve(opt_config const* cfg, opt_context* ctx, opt_state* state)
{
  // Initialize matrices needed for BFGS updates
  // M represents the inverse Hessian approximation
  dmatrix_t M = dmatrix_t::identity(ctx->n_params);
  dmatrix_t prev_grad = dmatrix_t::zero(ctx->n_params, 1);
  dmatrix_t prev_params = dmatrix_t::zero(ctx->n_params, 1);
  // Pre-compute identity matrix for BFGS update
  const dmatrix_t I_n = dmatrix_t::identity(ctx->n_params);
  // Initialize terms structure with only the masks needed for BFGS
  opt_iteration_terms terms(ctx->n_params, ctx->n_data, BFGS_TERMS_MASK);
  // Track previous loss for convergence detection
  float32_t prev_loss = INFINITY;
  // Temporary matrices we'll reuse throughout iterations to avoid allocations
  dmatrix_t s = dmatrix_t::zero(ctx->n_params, 1);             // Parameter difference
  dmatrix_t y = dmatrix_t::zero(ctx->n_params, 1);             // Gradient difference
  dmatrix_t A = dmatrix_t::zero(ctx->n_params, ctx->n_params); // BFGS update matrix
  for (state->iterations = 0; state->iterations < cfg->max_iterations; ++state->iterations)
  {
    // Compute iteration terms and update loss
    ctx->iteration_terms(terms, ctx->params, ctx->fit_data);
    state->loss = terms.loss;
    // Check for relative improvement after initial iterations
    if (state->iterations > 5 && ((prev_loss - state->loss) / prev_loss < 1e-3f))
    {
      state->termination_reason = OPT_CONVERGED;
      return;
    }
    prev_loss = state->loss;
    // Update inverse Hessian approximation after first iteration
    if (state->iterations > 0)
    {
      // Compute s = x_{k+1} - x_k and y = ∇f_{k+1} - ∇f_k
      // Reuse pre-allocated matrices s and y
      s = ctx->params - prev_params;
      y = terms.gradient - prev_grad;
      // Compute curvature condition sy = s^T y
      float32_t sy = s.dot(y);
      // Only update M if curvature condition is satisfied
      if (sy > 1e-10f)
      {
        float32_t rho = 1.0f / sy;
        // Compute BFGS update: M = (I - ρsy^T)M(I - ρys^T) + ρss^T
        // We break this into steps, reusing matrix A
        // First compute A = (I - ρsy^T)
        A = I_n - (rho * s * y.transpose());
        // Update M = AMA^T + ρss^T
        // Note: We do this in-place to minimize memory usage
        M = (A * M * A.transpose()) + (rho * s * s.transpose());
      }
    }
    // Store current state for next iteration
    prev_grad = terms.gradient;
    prev_params = ctx->params;
    clip_gradient_norm(terms.gradient, 2.0f);
    // Compute search direction: delta = -M∇f
    dmatrix_t delta = -1.0f * (M * terms.gradient);
    // Perform line search and scale the update
    float32_t alpha =
        wolfe_line_search(ctx, ctx->params, delta, terms.gradient, state->loss, 10, 1.0f, 0.004f, 0.9f, 0.8f);
    // float32_t alpha = 0.008f;
    delta *= alpha;
    // Check convergence based on step size
    if (delta.norm() < cfg->min_update_step)
    {
      state->termination_reason = OPT_CONVERGED;
      return;
    }
    // Update parameters
    ctx->params += delta;
  }
  state->loss = ctx->loss(ctx->params, ctx->fit_data);
  state->termination_reason = OPT_MAX_ITERATIONS_REACHED;
}
#endif
