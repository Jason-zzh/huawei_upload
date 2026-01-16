/**
 * Copyright 2025 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <string.h>
#include <torch/extension.h>
#include <iostream>
#include <vector>

#include "utils/op_utils.h"

namespace op_plugin {
namespace aten_op {
extern "C" int BatchNormGradExt(int nparam, void **params, int *ndims, int64_t **shapes, const char **dtypes, void *stream,
                                void *extra) {
  // Convert physical parameters passed by the framework to PyTorch tensor containers
  auto tensors = ConvertToATenTensors(nparam, params, ndims, shapes, dtypes, extra, c10::kCPU);

  // Extract input tensors (based on YAML: dout, input, weight, running_mean, running_var, saved_mean, saved_rstd)
  auto at_dout = tensors[0];           // gradient of output
  auto at_input = tensors[1];          // original input
  auto at_weight = tensors[2];         // weight parameter
  auto at_running_mean = tensors[3];   // running mean
  auto at_running_var = tensors[4];    // running variance
  auto at_saved_mean = tensors[5];     // saved mean from forward pass
  auto at_saved_rstd = tensors[6];     // saved rstd (reciprocal std) from forward pass

  // Get output tensors
  auto at_dx = tensors[nparam - 3];      // gradient w.r.t. input
  auto at_dweight = tensors[nparam - 2]; // gradient w.r.t. weight
  auto at_dbias = tensors[nparam - 1];   // gradient w.r.t. bias

  // Extract scalar parameters
  KernelInputInfo &input_info = *static_cast<KernelInputInfo *>(extra);
  KernelInputUtils input_utils(input_info);

  // Extract training and eps parameters
  bool training = input_utils.GetBoolInput(7);  // training parameter comes after 7 tensor inputs
  double eps = input_utils.GetFloatInput(8);    // eps parameter comes after training

  // For output_mask, we need to handle it as a tuple of integers (default (1, 1, 1))
  // This indicates which gradients to compute: (input_grad, weight_grad, bias_grad)
  std::array<bool, 3> output_mask = {true, true, true}; // By default, compute all gradients

  // Convert tensors to optional tensors for ATen
  c10::optional<at::Tensor> weight_opt = at_weight.defined() ? c10::optional<at::Tensor>(at_weight) : c10::nullopt;
  c10::optional<at::Tensor> running_mean_opt = at_running_mean.defined() ? c10::optional<at::Tensor>(at_running_mean) : c10::nullopt;
  c10::optional<at::Tensor> running_var_opt = at_running_var.defined() ? c10::optional<at::Tensor>(at_running_var) : c10::nullopt;
  c10::optional<at::Tensor> saved_mean_opt = at_saved_mean.defined() ? c10::optional<at::Tensor>(at_saved_mean) : c10::nullopt;
  c10::optional<at::Tensor> saved_rstd_opt = at_saved_rstd.defined() ? c10::optional<at::Tensor>(at_saved_rstd) : c10::nullopt;

  // Call ATen interface: native_batch_norm_backward_out
  at::_ops::native_batch_norm_backward_out::call(
    at_dout, at_input, weight_opt, running_mean_opt, running_var_opt,
    saved_mean_opt, saved_rstd_opt, training, eps, output_mask,
    at_dx, at_dweight, at_dbias);

  return 0;
}
}  // namespace aten_op
}  // namespace op_plugin
