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

#include "utils/op_utils.h"

namespace op_plugin {
namespace aten_op {
extern "C" int BatchNormExt(int nparam, void **params, int *ndims, int64_t **shapes, const char **dtypes, void *stream,
                            void *extra) {
  // Convert physical parameters passed by the framework to PyTorch tensor containers
  auto tensors = ConvertToATenTensors(nparam, params, ndims, shapes, dtypes, extra, c10::kCPU);

  // Extract input tensors (based on YAML: input, weight, bias, running_mean, running_var)
  auto at_input = tensors[0];
  auto at_weight = tensors[1];
  auto at_bias = tensors[2];
  auto at_running_mean = tensors[3];
  auto at_running_var = tensors[4];

  // Get output tensors (last 3 in the list: output, saved_mean, saved_variance)
  auto at_output = tensors[nparam - 3];      // output tensor
  auto at_saved_mean = tensors[nparam - 2];  // saved mean tensor
  auto at_saved_var = tensors[nparam - 1];   // saved invstd tensor (reciprocal of std)

  // Extract scalar parameters
  KernelInputInfo &input_info = *static_cast<KernelInputInfo *>(extra);
  KernelInputUtils input_utils(input_info);

  // Extract training, momentum, and epsilon parameters
  bool training = input_utils.GetBoolInput(5);  // training parameter comes after 5 tensor inputs
  double momentum = input_utils.GetFloatInput(6);  // momentum parameter comes after training
  double epsilon = input_utils.GetFloatInput(7);   // epsilon parameter comes after momentum

  // Convert tensors to optional tensors for ATen
  c10::optional<at::Tensor> weight_opt = at_weight.defined() ? c10::optional<at::Tensor>(at_weight) : c10::nullopt;
  c10::optional<at::Tensor> bias_opt = at_bias.defined() ? c10::optional<at::Tensor>(at_bias) : c10::nullopt;
  c10::optional<at::Tensor> running_mean_opt = at_running_mean.defined() ? c10::optional<at::Tensor>(at_running_mean) : c10::nullopt;
  c10::optional<at::Tensor> running_var_opt = at_running_var.defined() ? c10::optional<at::Tensor>(at_running_var) : c10::nullopt;

  // Call ATen interface: native_batch_norm_out
  at::_ops::native_batch_norm_out::call(
    at_input, weight_opt, bias_opt, running_mean_opt, running_var_opt, 
    training, momentum, epsilon, at_output, at_saved_mean, at_saved_var);

  return 0;
}
}  // namespace aten_op
}  // namespace op_plugin
