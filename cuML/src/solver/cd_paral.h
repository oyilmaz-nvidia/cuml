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

#include "ml_utils.h"
#include <cuda_utils.h>
#include <linalg/gemm.h>
#include <linalg/add.h>
#include <linalg/subtract.h>
#include <linalg/eltwise.h>
#include <linalg/unary_op.h>
#include <linalg/cublas_wrappers.h>
#include <matrix/math.h>
#include <matrix/matrix.h>
#include "glm/preprocess.h"
#include "shuffle.h"
#include <functions/penalty.h>
#include <functions/softThres.h>
#include <functions/linearReg.h>
#include <omp.h>

namespace ML {
namespace Solver {

using namespace MLCommon;

template<typename math_t>
void cdParalFit(math_t *input,
		   int n_rows,
		   int n_cols,
		   math_t *labels,
		   math_t *coef,
		   math_t *intercept,
		   bool fit_intercept,
		   bool normalize,
		   int epochs,
		   ML::loss_funct loss,
		   math_t alpha,
		   math_t l1_ratio,
		   math_t tol,
		   int n_processors,
		   cudaStream_t stream,
		   cublasHandle_t cublas_handle,
		   cusolverDnHandle_t cusolver_handle) {

	ASSERT(n_cols > 0,
			"Parameter n_cols: number of columns cannot be less than one");
	ASSERT(n_rows > 1,
			"Parameter n_rows: number of rows cannot be less than two");
	ASSERT(loss == ML::loss_funct::SQRD_LOSS,
			"Parameter loss: Only SQRT_LOSS function is supported for now");

	math_t *mu_input = NULL;
	math_t *mu_labels = NULL;
	math_t *norm2_input = NULL;
	math_t *preds = NULL;
	math_t *squared = NULL;
	math_t *loss_value = NULL;
	math_t *coef_copied = NULL;

	// Calculate here the first eigenvalue of AtA to get the number of processor (# of parallel operations)

	allocate(coef_copied, n_cols * n_processors, true);
	allocate(loss_value, 1);
	allocate(preds, n_rows * n_processors, true);
	allocate(squared, n_cols, true);

	std::vector<math_t> h_coef(n_cols, math_t(0));

	if (fit_intercept) {
		allocate(mu_input, n_cols);
		allocate(mu_labels, 1);

		GLM::preProcessData(input, n_rows, n_cols, labels, intercept, mu_input,
				mu_labels, norm2_input, fit_intercept, normalize, cublas_handle,
				cusolver_handle);
	}

	std::vector<int> rand_indices(n_cols);
	std::mt19937 g(rand());
	initShuffle(rand_indices, g);

	math_t l2_alpha = (1 - l1_ratio) * alpha * n_rows;
	alpha = l1_ratio * alpha * n_rows;

	if (normalize) {
		math_t scalar = math_t(1.0) + l2_alpha;
		Matrix::setValue(squared, squared, scalar, n_cols, stream);
	} else {
		LinAlg::colNorm(squared, input, n_cols, n_rows, LinAlg::L2Norm, false,
				stream);
		LinAlg::addScalar(squared, squared, l2_alpha, n_cols, stream);
	}

	for (int i = 0; i < epochs; i++) {
		if (i > 0) {
			Solver::shuffle(rand_indices, g);
		}

		math_t coef_max = 0.0;
		math_t d_coef_max = 0.0;
		math_t coef_prev = 0.0;

		for (int j = 0; j < n_processors; j++) {
			math_t *coef_copied_loc = coef_copied + (j * n_cols);
			Matrix::setValue(coef_copied_loc + rand_indices[j], coef_copied_loc + rand_indices[j], math_t(0.0), 1, stream);
		}

		LinAlg::gemm(input, n_rows, n_cols, coef_copied, preds, n_rows, n_processors,
							CUBLAS_OP_N, CUBLAS_OP_N, cublas_handle);

		Matrix::matrixVectorBinarySub(preds, labels, n_rows, n_processors, false, false);
		LinAlg::scalarMultiply(preds, preds, math_t(-1.0), n_rows * n_processors, stream);

		for (int j = 0; j < n_processors; j++) {
			int loc = j * n_rows;
			math_t *input_col_loc = input + (rand_indices[j] * n_rows);
			math_t *coef_loc = coef + rand_indices[j];
			math_t *squared_loc = squared + rand_indices[j];

			LinAlg::gemm(input_col_loc, n_rows, 1, preds + loc, coef_loc, 1, 1,
						CUBLAS_OP_T, CUBLAS_OP_N, cublas_handle);

			if (l1_ratio > math_t(0.0))
				Functions::softThres(coef_loc, coef_loc, alpha, 1);

			LinAlg::eltwiseDivideCheckZero(coef_loc, coef_loc, squared_loc, 1);

			coef_prev = h_coef[rand_indices[j]];
			updateHost(&(h_coef[rand_indices[j]]), coef_loc, 1);
			math_t diff = abs(coef_prev - h_coef[rand_indices[j]]);

			if (diff > d_coef_max)
				d_coef_max = diff;

			if (abs(h_coef[rand_indices[j]]) > coef_max)
				coef_max = abs(h_coef[rand_indices[j]]);
		}

		bool flag_continue = true;
		if (coef_max == math_t(0)) {
			flag_continue = false;
		}

		if ((d_coef_max / coef_max) < tol) {
			flag_continue = false;
		}

		if (!flag_continue) {
			//printf("iter:%d, coef_max: %f, d_coef_max: %f\n", i, coef_max, d_coef_max);
			break;
		}

		for (int j = 0; j < n_processors; j++) {
			math_t *coef_copied_loc = coef_copied + (j * n_cols);
			copy(coef_copied_loc, coef, n_cols);
		}
	}

	if (fit_intercept) {
		GLM::postProcessData(input, n_rows, n_cols, labels, coef, intercept,
				mu_input, mu_labels, norm2_input, fit_intercept, normalize,
				cublas_handle, cusolver_handle);

		if (mu_input != NULL)
			CUDA_CHECK(cudaFree(mu_input));
		if (mu_labels != NULL)
			CUDA_CHECK(cudaFree(mu_labels));
	} else {
		*intercept = math_t(0);
	}

	if (preds != NULL)
		CUDA_CHECK(cudaFree(preds));

	if (squared != NULL)
		CUDA_CHECK(cudaFree(squared));

	if (loss_value != NULL)
		CUDA_CHECK(cudaFree(loss_value));

	if (coef_copied != NULL)
		CUDA_CHECK(cudaFree(coef_copied));

}

/** @} */
}
;
}
;
// end namespace ML
