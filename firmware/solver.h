/*
 *      Author: Gokalp Cevik (gevik@wpi.edu)
 */

#ifndef INC_ICAS_SOLVER_SOLVER_H_
#define INC_ICAS_SOLVER_SOLVER_H_

#include "arm_math.h"
#include "icas/matrix.h"

#define SOLVER_ENABLE_LMA 0
#define SOLVER_ENABLE_GNA 1
#define SOLVER_ENABLE_GD 0
#define SOLVER_ENABLE_GD1O 0
#define SOLVER_ENABLE_ADAM 0
#define SOLVER_ENABLE_BFGS 0

// clang-format off
/*
 * Date: 1/16/2025, Author: Gokalp Cevik (gevik@wpi.edu)
 * This file contains the definitions for various optimizers for non-linear optimization (curve fitting). User is
 * expected to provide the relevant terms according to the specific problem (e.g. neural net training, lifetime fitting,
 * etc.). See icas/mono_exp_decay.h/cpp for a example implementation for the mono exponential decay problem.
 * For parameters the notation from the textbook, Numerical Optimization by Nocedal & Wright, is used. See the textbook
 * for more in-depth details. The choice of algorithm usually depends on several parameters including the number of
 * parameters, dataset size and the initial parameters.
 * If you have a well-behaved problem, small dataset (<1000), small number of parameters, and you can estimate the
 * initial conditions in a stable-manner, Gauss-Newton will outperform most.
 * If you have an ill-posed problem, small dataset (<1000), small number of parameters, and you can't estimate the
 * initial parameters, Levenberg-Marquardt or BFGS will be the most obvious choice. BFGS might converge faster
 * (superlinear), though it might become ill-posed depending on the problem.
 * 
 * For a frame of reference, asssume n parameters and m datapoints:
 * Levenberg-Marquardt: Converges linearly away from the solution, converges quadratically near the solution. Complexity = O((mn² + n³)log(1/ε))
 * Gauss-Newton: Converges quadratically near the solution, might become rank-deficient/unsolveable away from the solution. Complexity = O((mn² + n³)log(log(1/ε)))
 * BFGS: Converges superlinearly, usually stable. Complexity = O((mn + n²)log(1/ε))
 * Gradient Descent: Converges linearly, usually very stable but exhibits very slow convergence. Complexity = O(mnκ log(1/ε))
 * GD Momentum: O(mn√κ log(1/ε))
 * GD Adam: O(mn log(1/ε)) (but usually converges much much faster than regular or 1st order momentum gradient descent).
 * */
// clang-format on

// This flags indicate whether an algorithm needs a term explicitly.
static constexpr uint8_t COMPUTE_JACOBIAN_MASK = 1U << 0;
static constexpr uint8_t COMPUTE_RESIDUAL_MASK = 1U << 1;
static constexpr uint8_t COMPUTE_GRADIENT_MASK = 1U << 3;
static constexpr uint8_t COMPUTE_HESSIAN_MASK = 1U << 4;
static constexpr uint8_t COMPUTE_LOSS_MASK = 1U << 5;

// Levenberg-Marquardt requires the gradient and the approx. Hessian (GNA), and the loss at each iteration. We do not
// need to explicitly store the Jacobian nor the residual vector.
static constexpr uint8_t LEVENBERG_MARQUARDT_TERMS_MASK =
    COMPUTE_GRADIENT_MASK | COMPUTE_HESSIAN_MASK | COMPUTE_LOSS_MASK;
// Same case as Levenberg-Marquardt
static constexpr uint8_t GAUSS_NEWTON_TERMS_MASK = COMPUTE_GRADIENT_MASK | COMPUTE_HESSIAN_MASK | COMPUTE_LOSS_MASK;
// Gradient Descent and its variants (momentum, Adam) require only the gradient and the loss
static constexpr uint8_t GD_TERMS_MASK = COMPUTE_GRADIENT_MASK | COMPUTE_LOSS_MASK;
static constexpr uint8_t GD_MOMENTUM_TERMS_MASK = COMPUTE_GRADIENT_MASK | COMPUTE_LOSS_MASK;
static constexpr uint8_t GD_ADAM_TERMS_MASK = COMPUTE_GRADIENT_MASK | COMPUTE_LOSS_MASK;
// BFGS require only the gradient of the loss function, and approximates the Hessian (or its inverse through
// Sherman-Morrison-Woodburry formula) in each iteration through rank-two updates. We also compute the loss as the same
// for each algorithm.
static constexpr uint8_t BFGS_TERMS_MASK = COMPUTE_GRADIENT_MASK | COMPUTE_LOSS_MASK;

// Per iteration terms, some fields in this struct might not be available in each algorithm
struct opt_iteration_terms
{
  opt_iteration_terms(uint32_t n_params, uint32_t n_data, uint8_t mask)
  {
    // Allocate accordingly
    if (mask & COMPUTE_JACOBIAN_MASK)
    {
      J = dmatrix_t(n_data, n_params);
    }
    if (mask & COMPUTE_RESIDUAL_MASK)
    {
      r = dmatrix_t(n_data, 1);
    }
    if (mask & COMPUTE_GRADIENT_MASK)
    {
      gradient = dmatrix_t(n_params, 1);
    }
    if (mask & COMPUTE_HESSIAN_MASK)
    {
      H = dmatrix_t(n_params, n_params);
    }
  }
  dmatrix_t J;        // Jacobian
  dmatrix_t r;        // Residual vector
  dmatrix_t gradient; // Gradient of the loss function
  dmatrix_t H;        // Gauss-Newton approximation to the Hessian
  float32_t loss{};   // 1/2*r^T*r
};

using residual_fn = void (*)(const dmatrix_t&, dmatrix_t&, void*);
using Jacobian_fn = void (*)(const dmatrix_t&, dmatrix_t&, void*);
using gradient_fn = void (*)(const dmatrix_t&, dmatrix_t&, void*);
using loss_fn = float32_t (*)(const dmatrix_t&, void*);
using iteration_terms_fn = void (*)(opt_iteration_terms&, const dmatrix_t&, void*);

// Optimization algorithm
enum opt_algorithm
{
  OPT_LEVENBERG_MARQUARDT = 0, // Levenberg-Marquardt
  OPT_NEWTON_RAPHSON,          // Newton-Raphson (Not implemented)
  OPT_GAUSS_NEWTON,            // Gauss-Newton
  OPT_GRADIENT_DESCENT,        // Gradient Descent
  OPT_GD_MOMENTUM,             // Gradient Descent w/ 1st order momentum
  OPT_GD_ADAM,                 // Gradient Descent w/ 2nd order momentum (Adam)
  OPT_SGD,                     // Stochastic Gradient Descent (SGD)
  OPT_SGD_MOMENTUM,            // SGD w/ 1st order momentum
  OPT_SGD_ADAM,                // SGD w/ 2nd order momentum (Adam)
  OPT_BFGS,                    // Broyden–Fletcher–Goldfarb–Shanno (BFGS)
  OPT_LBFGS                    // Limited memory Broyden–Fletcher–Goldfarb–Shanno (L-BFGS) (Not implemented)
};

enum opt_termination_reason
{
  OPT_CONVERGED = 0,
  OPT_ZERO_GRADIENT,
  OPT_ILL_CONDITIONED,
  OPT_MAX_ITERATIONS_REACHED,
  OPT_EARLY_STOPPING // NOTE: Currently not used
};

char const* opt_termination_reason_to_string(opt_termination_reason reason);
char const* opt_algorithm_to_string(opt_algorithm algorithm);


#if SOLVER_ENABLE_LMA==1
struct opt_lma_config
{
  float32_t mean_diagonal_scaling_factor = 1e-3f;
  float32_t rho_good_th = 0.75f;           // Local quadratic approximation 'good' threshold
  float32_t rho_bad_th = 0.25f;            // Local quadratic approximation 'bad' threshold
  float32_t lambda_increase_factor = 2.0f; // Lambda increase factor when reduction rate is above the threshold
  float32_t lambda_decrease_factor = 0.5f; // Lambda decrease factor when reduction rate is below the threshold
  [[maybe_unused]] uint32_t broyden_reset_period = 8; // Determines how often to recompute the full Jacobian
};
#endif

#if SOLVER_ENABLE_GNA==1
struct opt_gna_config
{
  float32_t scaling = 1.0f;       // Fixed scaling factor for the detal
  float32_t damping = 1e-6f;      // Damping factor added to the augmented normal matrix
  float32_t initial_alpha = 1.0f; // Initial step length
};
#endif

#if SOLVER_ENABLE_GD1O==1
struct opt_gd1o_config
{
  float32_t mu = 0.94f; // First-order moment exponential decay rate
};
#endif

#if SOLVER_ENABLE_ADAM==1
struct opt_gd_adam_config
{
  float32_t mu = 0.94f;  // First-order moment exponential decay rate
  float32_t nu = 0.96f; // Second-order moment (AKA velocity or uncentered variance) exponential decay rate
};
#endif

struct opt_config
{
  opt_algorithm algorithm = OPT_LEVENBERG_MARQUARDT;
  /* Common parameters for every algorithm */
  int max_iterations =
      20; // Maximum allowed iterations, increase this significantly for slow-converging algorithms (gradient descent).
  float min_update_step = 5.0f * 1e-4f; // Minimum allowed update step, used in all algorithms to check for convergence
  [[maybe_unused]] float min_gradient_norm =
      1e-6f;                  // Minimum allowed gradient norm to continue solving, may not be used in all algorithms
  float min_diagonal = 1e-6f; // Used to check ill-conditioned normal matrices, may not be used in all algorithms
  float min_error_reduction = 1e-6f; // NOTE: May not be used
  /****************************************/
  void* algorithm_config = nullptr;
  void* line_search_config = nullptr;
};

struct opt_context
{
  explicit opt_context(uint32_t n_params) : n_params(n_params)
  {
    // Allocate the parameter vector in the constructor
    params = dmatrix_t(n_params, 1); // Vectors are always row vectors
  }
  dmatrix_t params;               // Parameter matrix
  uint32_t n_params;              // Number of parameters
  uint32_t n_data;                // Number of datapoints
  void* fit_data = nullptr;       // Fit data (dataset, # points)
  Jacobian_fn jacobian = nullptr; // Jacobian of the residuals
  residual_fn residual = nullptr; // Residual function
  gradient_fn gradient = nullptr; // Gradient of the loss (objective) function
  loss_fn loss = nullptr;
  iteration_terms_fn iteration_terms = nullptr; // Computes all iteration terms in one-go
};

struct opt_state
{
  int iterations = 0;
  float loss = 0.0f;
  int termination_reason = -1;
};

enum opt_status
{
  OPT_FAILURE = 0,
  OPT_SUCCESS
};

void opt_init_default_params(opt_algorithm algorithm);

#if SOLVER_ENABLE_LMA==1
void opt_levenberg_marquardt_solve(opt_config const* cfg, opt_context* ctx, opt_state* state);
#endif
#if SOLVER_ENABLE_GNA==1
void opt_gauss_newton_solve(opt_config const* cfg, opt_context* ctx, opt_state* state);
#endif
#if SOLVER_ENABLE_GD==1
void opt_gradient_descent_solve(opt_config const* cfg, opt_context* ctx, opt_state* state);
#endif
#if SOLVER_ENABLE_GD1O==1
void opt_gd_momentum_solve(opt_config const* cfg, opt_context* ctx, opt_state* state);
#endif
#if SOLVER_ENABLE_ADAM==1
void opt_gd_adam_solve(opt_config const* cfg, opt_context* ctx, opt_state* state);
#endif
#if SOLVER_ENABLE_BFGS==1
void opt_bfgs_solve(opt_config const* cfg, opt_context* ctx, opt_state* state);
#endif

#endif /* INC_ICAS_SOLVER_SOLVER_H_ */
