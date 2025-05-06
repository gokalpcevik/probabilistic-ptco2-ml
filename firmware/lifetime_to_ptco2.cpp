#include "icas/nn/lifetime_to_ptco2.h"
#include "icas/matrix.h"
#include <stdio.h>
#include <stdlib.h>

/*  NOTE: NOT WORKING
float32_t PtcO2_regressor_s8_t::predict_PtcO2(PtcO2_regressor_x* x)
{
  // Scratch buffer for intermediate activations
  static int8_t buffer_a[128]; // Max hidden layer size
  static int8_t buffer_b[128]; // Second buffer for ping-pong
  int8_t* input_ptr = buffer_a;
  int8_t* output_ptr = buffer_b;
  int8_t final_output;

#define SCRATCH_BUFFER_SIZE 1024
  static int32_t scratch_buffer[SCRATCH_BUFFER_SIZE / sizeof(int32_t)];
  cmsis_nn_context ctx;
  ctx.buf = scratch_buffer;
  ctx.size = SCRATCH_BUFFER_SIZE;

  // First, normalize and quantize the input features
  float normalized_input[3];
  normalize_input(x->tensor, normalized_input);

  // Convert to quantized int8
  int8_t quantized_input[3];
  quantize_input(normalized_input, quantized_input);

  // DEBUG: Print input values
  printf("\r\n===== DEBUG =====\r\n");
  printf("Raw input: %f, %f, %f\r\n", x->tensor[0], x->tensor[1], x->tensor[2]);
  printf("Normalized input: %f, %f, %f\r\n", normalized_input[0], normalized_input[1], normalized_input[2]);
  printf("Quantized input (int8): %d, %d, %d\r\n", quantized_input[0], quantized_input[1], quantized_input[2]);

  // Copy to first input buffer
  memcpy(input_ptr, quantized_input, 3 * sizeof(int8_t));

  // Process each layer in sequence
  int8_t* temp_ptr;

  // Layer dimensions for all layers
  cmsis_nn_dims input_dims, filter_dims, bias_dims, output_dims;

  // Common settings for all layers
  input_dims.n = 1; // Batch size is 1 (single inference)
  input_dims.h = 1; // Height is 1 for MLPs (no spatial dimensions)
  input_dims.w = 1; // Width is 1 for MLPs (no spatial dimensions)

  filter_dims.h = 1; // Height is 1 for MLPs
  filter_dims.w = 1; // Width is 1 for MLPs

  bias_dims.n = 1; // Always 1 for bias
  bias_dims.h = 1;
  bias_dims.w = 1;
  bias_dims.c = 1;

  output_dims.n = 1; // Batch size
  output_dims.h = 1; // Height (1 for MLP)
  output_dims.w = 1; // Width (1 for MLP)

  // ------- Layer 0: 3 -> 128 -------
  input_dims.c = 3;    // Input features
  filter_dims.n = 3;   // INPUT feature dimension
  filter_dims.c = 128; // Not used for computations
  output_dims.c = 128; // OUTPUT feature dimension
                       //
  printf("\r\nLayer 0 (3->128):\r\n");
  printf("Input values: %d, %d, %d\r\n", input_ptr[0], input_ptr[1], input_ptr[2]);

  arm_fully_connected_per_channel_s8(&ctx, &L0_FC_PARAMS, &L0_PER_CHANNEL_PARAMS, &input_dims, input_ptr, &filter_dims,
                                     L0_WEIGHTS, &bias_dims, L0_BIASES, &output_dims, output_ptr);

  // Debug: Print first few output values
  printf("Layer 0 output (first 16): ");
  for (int i = 0; i < 16; i++)
  {
    printf("%d ", output_ptr[i]);
  }
  printf("...\r\n");

  // Swap buffers
  temp_ptr = input_ptr;
  input_ptr = output_ptr;
  output_ptr = temp_ptr;

  // ------- Layer 1: 128 -> 64 -------
  input_dims.c = 128;  // Input features
  filter_dims.n = 128; // INPUT feature dimension
  filter_dims.c = 64;  // Not used for computations
  output_dims.c = 64;  // OUTPUT feature dimension

  printf("\r\nLayer 1 (128->64):\r\n");
  printf("Input values (first 16): ");
  for (int i = 0; i < 16; i++)
  {
    printf("%d ", input_ptr[i]);
  }
  printf("...\r\n");

  arm_fully_connected_per_channel_s8(&ctx, &L1_FC_PARAMS, &L1_PER_CHANNEL_PARAMS, &input_dims, input_ptr, &filter_dims,
                                     L1_WEIGHTS, &bias_dims, L1_BIASES, &output_dims, output_ptr);

  // Debug: Print first few output values
  printf("Layer 1 output (first 16): ");
  for (int i = 0; i < 16; i++)
  {
    printf("%d ", output_ptr[i]);
  }
  printf("...\r\n");

  // Swap buffers
  temp_ptr = input_ptr;
  input_ptr = output_ptr;
  output_ptr = temp_ptr;

  // ------- Layer 2: 64 -> 32 -------
  input_dims.c = 64;  // Input features
  filter_dims.n = 64; // INPUT feature dimension
  filter_dims.c = 32; // Not used for computations
  output_dims.c = 32; // OUTPUT feature dimension

  printf("\r\nLayer 2 (64->32):\r\n");
  printf("Input values (first 16): ");
  for (int i = 0; i < 16; i++)
  {
    printf("%d ", input_ptr[i]);
  }
  printf("...\r\n");

  arm_fully_connected_per_channel_s8(&ctx, &L2_FC_PARAMS, &L2_PER_CHANNEL_PARAMS, &input_dims, input_ptr, &filter_dims,
                                     L2_WEIGHTS, &bias_dims, L2_BIASES, &output_dims, output_ptr);

  // Debug: Print first few output values
  printf("Layer 2 output (first 8): ");
  for (int i = 0; i < 8; i++)
  {
    printf("%d ", output_ptr[i]);
  }
  printf("...\r\n");

  // Swap buffers
  temp_ptr = input_ptr;
  input_ptr = output_ptr;
  output_ptr = temp_ptr;

  // ------- Layer 3: 32 -> 16 -------
  input_dims.c = 32;  // Input features
  filter_dims.n = 32; // INPUT feature dimension
  filter_dims.c = 16; // Not used for computations
  output_dims.c = 16; // OUTPUT feature dimension

  printf("\r\nLayer 3 (32->16):\r\n");
  printf("Input values (first 8): ");
  for (int i = 0; i < 8; i++)
  {
    printf("%d ", input_ptr[i]);
  }
  printf("...\r\n");

  arm_fully_connected_per_channel_s8(&ctx, &L3_FC_PARAMS, &L3_PER_CHANNEL_PARAMS, &input_dims, input_ptr, &filter_dims,
                                     L3_WEIGHTS, &bias_dims, L3_BIASES, &output_dims, output_ptr);

  // Debug: Print first few output values
  printf("Layer 3 output (first 8): ");
  for (int i = 0; i < 8; i++)
  {
    printf("%d ", output_ptr[i]);
  }
  printf("...\r\n");

  // Swap buffers
  temp_ptr = input_ptr;
  input_ptr = output_ptr;
  output_ptr = temp_ptr;

  // ------- Layer 4: 16 -> 1 (Final output) -------
  input_dims.c = 16;  // Input features
  filter_dims.n = 16; // INPUT feature dimension
  filter_dims.c = 1;  // Not used for computations
  output_dims.c = 1;  // OUTPUT feature dimension

  printf("\r\nLayer 4 (16->1):\r\n");
  printf("Input values (all 16): ");
  for (int i = 0; i < 16; i++)
  {
    printf("%d ", input_ptr[i]);
  }
  printf("\r\n");

  arm_fully_connected_per_channel_s8(&ctx, &L4_FC_PARAMS, &L4_PER_CHANNEL_PARAMS, &input_dims, input_ptr, &filter_dims,
                                     L4_WEIGHTS, &bias_dims, L4_BIASES, &output_dims, output_ptr);

  // Debug: Print final quantized output
  printf("Layer 4 output (final): %d\r\n", output_ptr[0]);

  // Dequantize the final result
  final_output = output_ptr[0];
  float PtcO2_result = dequantize_output(final_output);

  printf("Final dequantized result: %f\r\n", PtcO2_result);
  printf("===== END DEBUG =====\r\n\r\n");

  return PtcO2_result;
}

*/

extern "C"
{
#include "ff.h" // FatFS
}

float32_t PtcO2_regressor_s8_t::predict_PtcO2(PtcO2_regressor_x* x)
{
  FIL fw, fb, f_as, f_ws, f_os, f_zp;
  UINT br;

  // Open all model parameter files
  if (f_open(&fw, "models/weights_q8.bin", FA_READ) != FR_OK || f_open(&fb, "models/bias_q8.bin", FA_READ) != FR_OK ||
      f_open(&f_as, "models/act_scales.bin", FA_READ) != FR_OK ||
      f_open(&f_ws, "models/weight_scales.bin", FA_READ) != FR_OK ||
      f_open(&f_os, "models/output_scales.bin", FA_READ) != FR_OK ||
      f_open(&f_zp, "models/zero_points_act.bin", FA_READ) != FR_OK)
  {
    printf("[Q8] Failed to open one or more model files.\r\n");
    return -999.0f;
  }

  const int layers[5][2] = {{128, 3}, {64, 128}, {32, 64}, {16, 32}, {1, 16}};

  uint8_t input[128]; // input and intermediate activations
  uint8_t output[128];
  int8_t weights_row[128];
  int32_t bias_val;
  float act_scale, out_scale;
  uint8_t act_zp, out_zp;

  // Initial input
  float norm[3];
  normalize_input(x->tensor, norm);
  quantize_input(norm, input); // result is uint8_t[3] in [0, 127]

  uint32_t w_offset = 0;
  uint32_t b_offset = 0;
  uint32_t ws_offset = 0;

  for (int layer = 0; layer < 5; ++layer)
  {
    const int out_dim = layers[layer][0];
    const int in_dim = layers[layer][1];

    // Read layer-wide activation scale and output scale
    f_read(&f_as, &act_scale, sizeof(float), &br);
    f_read(&f_os, &out_scale, sizeof(float), &br);
    f_read(&f_zp, &act_zp, sizeof(uint8_t), &br); // input zero-point
    f_read(&f_zp, &out_zp, sizeof(uint8_t), &br); // output zero-point

    for (int i = 0; i < out_dim; ++i)
    {
      f_lseek(&fw, w_offset + i * in_dim);
      f_read(&fw, weights_row, in_dim, &br);
      f_lseek(&fb, b_offset + i * sizeof(int32_t));
      f_read(&fb, &bias_val, sizeof(int32_t), &br);
      float w_scale;
      f_lseek(&f_ws, ws_offset + i * sizeof(float));
      f_read(&f_ws, &w_scale, sizeof(float), &br);
      int32_t acc = bias_val;
      for (int j = 0; j < in_dim; ++j)
        acc += static_cast<int32_t>(weights_row[j]) * (static_cast<int32_t>(input[j]) - act_zp);
      // y_q = clamp(round(acc * act_scale / (w_scale * out_scale)) + out_zp, 0, 127)
      float scale = act_scale / (w_scale * out_scale);
      int32_t y_q = static_cast<int32_t>(roundf(acc * scale)) + out_zp;
      y_q = fminf(127, fmaxf(0, y_q));
      output[i] = static_cast<uint8_t>(y_q);
    }
    memcpy(input, output, out_dim);
    w_offset += out_dim * in_dim;
    b_offset += out_dim * sizeof(int32_t);
    ws_offset += out_dim * sizeof(float);
  }

  // Final dequantized output
  float final_output = (static_cast<int32_t>(output[0]) - out_zp) / out_scale;

  // Close all files
  f_close(&fw);
  f_close(&fb);
  f_close(&f_as);
  f_close(&f_ws);
  f_close(&f_os);
  f_close(&f_zp);
  return final_output;
}

float32_t PtcO2_regressor_t::predict_PtcO2(PtcO2_regressor_x* x)
{
  FIL fw, fb;
  UINT br;
  FRESULT res;
  res = f_open(&fw, "models/w.bin", FA_READ);
  if (res != FR_OK)
  {
    printf("[PtcO2] Failed to open w.bin\r\n");
    return -999.0f;
  }
  res = f_open(&fb, "models/b.bin", FA_READ);
  if (res != FR_OK)
  {
    printf("[PtcO2] Failed to open b.bin\r\n");
    f_close(&fw);
    return -999.0f;
  }
  float input[128];       // max input size
  float output[128];      // reused between layers
  float weights_row[128]; // temporary buffer for one row of weights
  float bias_val;         // one bias
  normalize_input(x->tensor, input);
  const int layers[5][2] = {{128, 3}, {64, 128}, {32, 64}, {16, 32}, {1, 16}};
  uint32_t weight_offset = 0;
  uint32_t bias_offset = 0;
  for (int layer = 0; layer < 5; ++layer)
  {
    const int out_dim = layers[layer][0];
    const int in_dim = layers[layer][1];
    for (int i = 0; i < out_dim; ++i)
    {
      f_lseek(&fw, weight_offset + i * in_dim * sizeof(float));
      if (f_read(&fw, weights_row, sizeof(float) * in_dim, &br) != FR_OK || br != sizeof(float) * in_dim)
      {
        printf("[PtcO2] Weight read fail at layer %d neuron %d\r\n", layer, i);
        goto fail;
      }
      f_lseek(&fb, bias_offset + i * sizeof(float));
      if (f_read(&fb, &bias_val, sizeof(float), &br) != FR_OK || br != sizeof(float))
      {
        printf("[PtcO2] Bias read fail at layer %d neuron %d\r\n", layer, i);
        goto fail;
      }
      float sum = bias_val;
      for (int j = 0; j < in_dim; ++j)
        sum += weights_row[j] * input[j];
      output[i] = (layer < 4) ? std::max(sum, 0.0f) : sum;
    }
    memcpy(input, output, sizeof(float) * out_dim);
    weight_offset += out_dim * in_dim * sizeof(float);
    bias_offset += out_dim * sizeof(float);
  }
  f_close(&fw);
  f_close(&fb);
  return input[0];

fail:
  f_close(&fw);
  f_close(&fb);
  return -999.0f;
}
