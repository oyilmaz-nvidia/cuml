/*
 * Copyright (c) 2018, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <cuda_utils.h>
#include <linalg/cublas_wrappers.h>
#include <linalg/eig.h>
#include <linalg/eltwise.h>
#include <linalg/transpose.h>
#include <matrix/math.h>
#include <matrix/matrix.h>
#include <stats/cov.h>
#include <stats/mean.h>
#include <stats/mean_center.h>
#include <cumlprims.hpp>
#include "common/cumlHandle.hpp"
#include "common/device_buffer.hpp"
#include "cuML.hpp"
#include "ml_utils.h"
#include "tsvd/tsvd.h"

namespace ML {

using namespace MLCommon;

template <typename math_t>
void pcaFitTransformOpg(const cumlHandle_impl &handle, math_t *input,
        math_t *trans_input, math_t *components,
        math_t *explained_var, math_t *explained_var_ratio,
        math_t *singular_vals, math_t *mu, math_t *noise_vars,
        paramsPCA prms, cudaStream_t stream) {
	printf("pcaFitOpg called\n");
}

};  // end namespace ML
