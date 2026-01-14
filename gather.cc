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
extern "C" int Gather(int nparam, void **params, int *ndims, int64_t **shapes, const char **dtypes, void *stream,
                      void *extra) {
  auto tensors = ConvertToATenTensors(nparam, params, ndims, shapes, dtypes, extra, c10::kCPU);

  KernelInputInfo &input_info = *static_cast<KernelInputInfo *>(extra);
  KernelInputUtils input_utils(input_info);

  // Parameters: self, dim, index, sparse_grad, output
  // The last tensor is the output tensor
  at::Tensor self = tensors[0];
  int64_t dim = input_utils.GetIntInput(1);  // dim is passed as scalar input
  at::Tensor index = tensors[2];
  bool sparse_grad = input_utils.GetBoolInput(3);  // sparse_grad is passed as scalar input
  at::Tensor output = tensors[nparam - 1];  // last tensor is output

  at::gather_out(output, self, dim, index, sparse_grad);
  return 0;
}
}  // namespace aten_op
}  // namespace op_plugin
