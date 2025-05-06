# Project Structure: PINNs for Oxygen Sensing

This project is organized into several directories and files to facilitate the prediction of the partial pressure of oxygen (pO₂) using Physics-Informed Neural Networks (PINNs). Below is a detailed breakdown of the project structure:


## `src/` Directory
The `src/` directory contains all the source code for the project. It is further divided into subdirectories based on functionality:

### `src/ml/`
This directory contains machine learning scripts for training, testing, and evaluating models.

- **Training Scripts**:
  - `train_multisite.py`: Trains a model for multisite oxygen sensing.
  - `train_sternvolmer.py`: Implements training based on the Stern-Volmer equation.
  - `train_msv.py`: Trains a model using multisite Stern-Volmer (MSV) data.
  - `train_joint.py`: Jointly trains models using multiple datasets.
  - `quant_two_stage.py`: Implements a two-stage quantification process.

- **Testing Scripts**:
  - `test_two_stage.py`: Tests the two-stage quantification model.
  - `test_multisite.py`: Evaluates the multisite model.
  - `test_hard_pinn.py`: Tests a hard-constrained PINN model.
  - `test_joint.py`: Evaluates the joint training model.

- **Utilities**:
  - `compare_ts_fp32_ptq.py`: Compares floating-point and quantized models.
  - `plot_dataset.py`: Visualizes datasets for analysis.

- **Models**:
  - `pO2regressor.py`: Defines the machine learning model for pO₂ regression.

### `src/data_processing/`
This directory contains scripts for data preprocessing and transformation.

- **Data Processing Scripts**:
  - `o2_dataset.py`: Handles dataset loading and preprocessing.
  - `plot_curves.py`: Plots curves for data visualization.
  - `log_to_data.py`: Converts log files into structured data.
  - `log_to_data_field.py`: Processes field-specific log data (updated version).
  
## `firmware/` Directory

Since the TOM firmware is a property of the ICAS Research Group and contains a few NDA requiring libraries, I am not able to share the full source code. I will however be explaining it in detail below and sharing my portion of the code necessary for this project. 

### Saving the Parameters to On-Board External Flash

After exporting the model parameters (fp32 or int8), I integrated them into the firmware development environment. I then run the following piece of code at MCU startup to upload the weights to the flash. Internally, a disk driver for the on-board external flash runs, and FatFs library handles formatting, saving, writing of the disk accordingly.

Below example is for the fp32 model, int8 code is a bit longer has more functionality (e.g. adjusting the scales, etc.) so I will not include it here. The idea is the same however.

```cpp
#ifdef WRITE_PARAMETERS

  {
    FIL fw;
    FIL fb;
    UINT bw;
    // ─────────────────────────────────────────────
    // Open files for writing
    res = f_open(&fw, "0:models/w.bin", FA_CREATE_ALWAYS | FA_WRITE);
    if (res != FR_OK && res != FR_EXIST)
    {
      printf("[PtcO2] Failed to create w.bin (%d)\r\n", res);
      enterLPMode(lp_mode);
    }

    res = f_open(&fb, "0:models/b.bin", FA_CREATE_ALWAYS | FA_WRITE);
    if (res != FR_OK && res != FR_EXIST)
    {
      printf("[PtcO2] Failed to create b.bin (%d)\r\n", res);
      f_close(&fw);
      enterLPMode(lp_mode);
    }

// ─────────────────────────────────────────────
#define WRITE_ARRAY(file, array, count)                                                                                \
  do                                                                                                                   \
  {                                                                                                                    \
    res = f_write(file, array, sizeof(float32_t) * (count), &bw);                                                      \
    if (res != FR_OK || bw != sizeof(float32_t) * (count))                                                             \
    {                                                                                                                  \
      printf("[PtcO2] Write failed: " #array ": %d\r\n", res);                                                         \
      f_close(&fw);                                                                                                    \
      f_close(&fb);                                                                                                    \
      enterLPMode(lp_mode);                                                                                            \
    }                                                                                                                  \
  } while (0)

    // ─────────────────────────────────────────────
    // Write weights
    WRITE_ARRAY(&fw, L0_WEIGHTS, 128 * 3);
    WRITE_ARRAY(&fw, L1_WEIGHTS, 64 * 128);
    WRITE_ARRAY(&fw, L2_WEIGHTS, 32 * 64);
    WRITE_ARRAY(&fw, L3_WEIGHTS, 16 * 32);
    WRITE_ARRAY(&fw, L4_WEIGHTS, 1 * 16);

    // ─────────────────────────────────────────────
    // Write biases
    WRITE_ARRAY(&fb, L0_BIASES, 128);
    WRITE_ARRAY(&fb, L1_BIASES, 64);
    WRITE_ARRAY(&fb, L2_BIASES, 32);
    WRITE_ARRAY(&fb, L3_BIASES, 16);
    WRITE_ARRAY(&fb, L4_BIASES, 1);

    // ─────────────────────────────────────────────
    // Cleanup
    f_close(&fw);
    f_close(&fb);
    printf("[PtcO2] Parameters written to flash.\r\n");
    return 0;
  }
#endif // WRITE_PARAMETERS
```

### The Inference Module

Initially, I tried to use CMSIS-NN for the inference on the MCU. However, after spending a ridicilously long amount of time not getting it to work, I've decided to keep the implementation simple and wrote a custom module. 

The inference module (containing both CMSIS-NN (not working) and the custom modules are located in `firmware/lifetime_to_ptco2.h/cpp.).

### Computing the Lifetime and Inference Process

The lifetime is computed by fitting a mono-exponential line of the form $I(t)=A + B * \exp(t/-\tau)$, where $\tau$ is the luminescence lifetime.

In the firmware, the analog front-end captures the luminescence decay curve internally and passes it to the microcontroller via I2C. For a single measurement, 10 decay curves are captured and averaged to reduce noise effects on the lifetime calculations. A single decay curve has 60 datapoints in total.

After acquiring the decay curve, the datapoints (intensity, timestamps) are normalized and passed on the the solver modules (`firmware/solver.h/cpp`, and `firmware/mono_exp_decay.h`). For this project and the demo, I am using the Gauss-Newton optimizer to solve for the lifetime. 

The solver module has more algorithms like Gradient Descent (with Momentum and Adam optimizers), line search functionality (Armijo or Wolfe conditions), BFGS and finally Levenberg-Marquardt optimizers.

The following code runs for each lifetime measurement and for each inferred PtcO$_2$:

```cpp
opt_context tau_ctx(3); // Optimization context with three parameters
opt_config tau_cfg{};
// opt_lma_config lma_cfg{};
opt_gna_config gna_cfg{};

// Lifetime calculation via non-linear optimization
tau_cfg.algorithm = OPT_GAUSS_NEWTON;
tau_cfg.algorithm_config = &gna_cfg;
tau_cfg.max_iterations = 25;
tau_cfg.min_update_step = 1e-4f;
// Setup the optimizer context
tau_ctx.iteration_terms = &mono_exp_iteration_terms_optimized_T<GAUSS_NEWTON_TERMS_MASK>;
tau_ctx.residual = &mono_exp_residuals;
tau_ctx.jacobian = &mono_exp_jacobian;
tau_ctx.gradient = &mono_exp_gradient;
tau_ctx.loss = &mono_exp_loss;
tau_ctx.fit_data = &exp_decay_data;
tau_ctx.n_data = NUM_INT;
tau_ctx.n_params = 3;
// Ambient air parameters
tau_ctx.params.at(0) = 0.2223f;
tau_ctx.params.at(1) = 0.5223f;
tau_ctx.params.at(2) = -3.131f;

// Since timestep does not change, we can precompute the timepoints of the decay curve
for (int i = 0; i < NUM_INT; i++) {
    exp_decay_data.time[i] = (time_step_us * (float)i) / O2_TIME_SCALE;
}

// Then in the main loop
while (measurements_active) {
    // .... setup AFE, acquire decay curve, etc.

    /* Lifetime fitting */
    opt_state tau_state;
    // The first measurement starts off away from the local minima, we solve for 5 times so it can properly catch up (?)
    if (numMeasurementsCntr == 0) {
        size_t initial_solver_steps = 300;
        float32_t lifetime[initial_solver_steps];
        tau_cfg.max_iterations = 300;
        tau_cfg.min_update_step = 1e-5f;
        printf("> First measurement -> solving for the lifetime will take a bit longer.\r\n");
        for (int i = 0; i < initial_solver_steps; ++i) {
            opt_gauss_newton_solve(&tau_cfg, &tau_ctx, &tau_state);
            lifetime[i] = -O2_TIME_SCALE / tau_ctx.params.at(2);
            printf("\033[32m> [Iteration %d] [us]: %.3f\033[0m\r\n", i + 1, lifetime[i]);
        }
        tau_cfg.max_iterations = 70;
    }
    opt_gauss_newton_solve(&tau_cfg, &tau_ctx, &tau_state);
    float lifetime = -O2_TIME_SCALE / tau_ctx.params.at(2);
    datapoint.lifetime = lifetime;
    printf("\033[31m- Optimizer status: \033[0m%s\r\n",
           opt_termination_reason_to_string((opt_termination_reason)tau_state.termination_reason));
    printf("\033[33m- Total iterations: \033[0m%u\r\n", tau_state.iterations);
    printf("\033[34m- Loss [a.u.]: \033[0m%f\r\n", tau_state.loss);
    printf("\033[92m- Lifetime [us]: \033[0m%f\r\n", lifetime);
    printf("\033[36m- Measurement timestamp [ms]: \033[0m%d \r\n", datapoint.ts);
    /* ~~Lifetime */

    // Rest of the sensor functionality...

    // Predict PtcO2
    /* PtcO2 Prediction */
    PtcO2_regressor_x x;
    float32_t tau_avg = update_centered_moving_average(tau_cma_calc, lifetime);
    printf("> Lifetime (MA): %.3f\r\n", tau_avg);
    // I was using tau_avg before, but it seems to lag behind too much for accurate ptco2 predictions.
    x.tensor[0] = lifetime; // Lifetime
    x.tensor[1] = fTemp;    // Temperature
    x.tensor[2] = log1pf((float32_t)active_film.num_pulses_applied); // log1p(pulses)
    printf("> Starting inference.\r\n");
    reset_profiler(&inference_profiler);
    float32_t calibration_constant = 1.0f; // For future calibration methods
    float32_t ptco2 = PtcO2_regressor_s8_t::predict_PtcO2(&x) / calibration_constant;
    int64_t inference_elapsed = profiler_get_elapsed_ms(&inference_profiler);
    printf("> Inference finished.\r\n");
    printf("\033[36m- Inference elapsed time [ms] \033[0m%d\r\n", inference_elapsed);
    printf("\033[36m- Predicted PtcO2 [cmHg]: \033[0m%.4f \r\n", ptco2);

    // Save measurement data, exit if maximum number of measurements are reached...
}
```

## `data/` Directory
This directory contains raw and processed data used for training and evaluation.

- **Subdirectories**:
  - `raw/`: Stores raw data files.
  - `processed/`: Contains preprocessed datasets ready for use.

## `outputs/` Directory
This directory stores the outputs generated by the project.

- **Subdirectories**:
  - `mcu/`: Contains generated headers for the MCU implementation.
  - `plots/`: Stores generated plots and visualizations.
  - `images/`: Contains images for the report.

# How to Train and See the Results

If you are on Windows, you can just run `.\scripts\train_all.bat` to train all the models.

If you are on Mac/Linux:

Classic Stern-Volmer Model (1919)
`python -m src.ml.training.train_sternvolmer`

Modified Stern-Volmer Model
`python -m src.ml.training.train_msv`

Multi-site Quenching Model
`python -m src.ml.training.train_multisite`

Joint Regressor
`python -m src.ml.training.train_pinn_extrap`

Two-Stage Regressor
`python -m src.ml.training.train_two_stage --stage 2 --lambda_phys 25 --unfreeze_pct 0.9`

To see the results, call the plotting modules with no arguments.


### Quantization Model - Testing Simulated/Artifical Inputs

You can apply simulated or user-set inputs to the quantized model by running the following module with the arguments:

`python -m src.ml.q_test_x <lifetime_us> <temperature> <pulses_applied>`

Arguments are normalized in the module, so you don't need to normalize them manually in the arguments.

---