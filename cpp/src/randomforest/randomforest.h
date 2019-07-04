/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
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
#include <map>
#include "decisiontree/decisiontree.h"

namespace ML {

enum RF_type {
  CLASSIFICATION,
  REGRESSION,
};

struct RF_metrics {
  RF_type rf_type;

  // Classification metrics
  float accuracy = -1.0f;

  // Regression metrics
  double mean_abs_error = -1.0;
  double mean_squared_error = -1.0;
  double median_abs_error = -1.0;

  RF_metrics(float cfg_accuracy);
  RF_metrics(double cfg_mean_abs_error, double cfg_mean_squared_error,
             double cfg_median_abs_error);
  void print();
};

struct RF_params {
  /**
   * Control bootstrapping. If set, each tree in the forest is built on a bootstrapped sample with replacement.
   * If false, sampling without replacement is done.
   */
  bool bootstrap = true;
  /**
   * Control bootstrapping for features. If features are drawn with or without replacement
   */
  bool bootstrap_features = false;
  /**
   * Number of decision trees in the random forest.
   */
  int n_trees;
  /**
   * Ratio of dataset rows used while fitting each tree.
   */
  float rows_sample = 1.0f;
  /**
   * Decision tree training hyper parameter struct.
   */
  DecisionTree::DecisionTreeParams tree_params;

  RF_params();
  RF_params(int cfg_n_trees);
  RF_params(bool cfg_bootstrap, bool cfg_bootstrap_features, int cfg_n_trees,
            float cfg_rows_sample);
  RF_params(bool cfg_bootstrap, bool cfg_bootstrap_features, int cfg_n_trees,
            float cfg_rows_sample,
            DecisionTree::DecisionTreeParams cfg_tree_params);
  void validity_check() const;
  void print() const;
};

/* Update labels so they are unique from 0 to n_unique_vals.
   Create an old_label to new_label map per random forest.
*/
void preprocess_labels(int n_rows, std::vector<int>& labels,
                       std::map<int, int>& labels_map, bool verbose = false);

/* Revert preprocessing effect, if needed. */
void postprocess_labels(int n_rows, std::vector<int>& labels,
                        std::map<int, int>& labels_map, bool verbose = false);

template <class T, class L>
class rf {
 protected:
  RF_params rf_params;
  int rf_type;
  virtual const DecisionTree::DecisionTreeBase<T, L>* get_trees_ptr() const = 0;
  virtual ~rf() = default;
  void prepare_fit_per_tree(const ML::cumlHandle_impl& handle, int tree_id,
                            int n_rows, int n_sampled_rows,
                            unsigned int* selected_rows,
                            unsigned int* sorted_selected_rows,
                            char* rows_temp_storage, size_t temp_storage_bytes);

  void error_checking(const T* input, L* predictions, int n_rows, int n_cols,
                      bool is_predict) const;

 public:
  rf(RF_params cfg_rf_params, int cfg_rf_type = RF_type::CLASSIFICATION);

  int get_ntrees();
  void print_rf_summary();
  void print_rf_detailed();
};

template <class T>
class rfClassifier : public rf<T, int> {
 private:
  DecisionTree::DecisionTreeClassifier<T>* trees = nullptr;
  const DecisionTree::DecisionTreeClassifier<T>* get_trees_ptr() const;

 public:
  rfClassifier(RF_params cfg_rf_params);
  ~rfClassifier();

  void fit(const cumlHandle& user_handle, T* input, int n_rows, int n_cols,
           int* labels, int n_unique_labels);
  void predict(const cumlHandle& user_handle, const T* input, int n_rows,
               int n_cols, int* predictions, bool verbose = false) const;

  RF_metrics score(const cumlHandle& user_handle, const T* input,
                   const int* ref_labels, int n_rows, int n_cols,
                   int* predictions, bool verbose = false) const;

  void predictAllDTs(const cumlHandle& user_handle, const T* input, int n_rows,
                     int n_cols, int* predictions, bool verbose = false) const;
};

template <class T>
class rfRegressor : public rf<T, T> {
 private:
  DecisionTree::DecisionTreeRegressor<T>* trees = nullptr;
  const DecisionTree::DecisionTreeRegressor<T>* get_trees_ptr() const;

 public:
  rfRegressor(RF_params cfg_rf_params);
  ~rfRegressor();

  void fit(const cumlHandle& user_handle, T* input, int n_rows, int n_cols,
           T* labels);
  void predict(const cumlHandle& user_handle, const T* input, int n_rows,
               int n_cols, T* predictions, bool verbose = false) const;
  RF_metrics score(const cumlHandle& user_handle, const T* input,
                   const T* ref_labels, int n_rows, int n_cols,
                   T* predictions, bool verbose = false) const;
};

// Stateless API functions: fit, predict and score.

// ----------------------------- Classification ----------------------------------- //

void fit(const cumlHandle& user_handle, rfClassifier<float>* rf_classifier,
         float* input, int n_rows, int n_cols, int* labels,
         int n_unique_labels);
void fit(const cumlHandle& user_handle, rfClassifier<double>* rf_classifier,
         double* input, int n_rows, int n_cols, int* labels,
         int n_unique_labels);

void predict(const cumlHandle& user_handle,
             const rfClassifier<float>* rf_classifier, const float* input,
             int n_rows, int n_cols, int* predictions, bool verbose = false);
void predict(const cumlHandle& user_handle,
             const rfClassifier<double>* rf_classifier, const double* input,
             int n_rows, int n_cols, int* predictions, bool verbose = false);

RF_metrics score(const cumlHandle& user_handle,
                 const rfClassifier<float>* rf_classifier, const float* input,
                 const int* ref_labels, int n_rows, int n_cols,
                 int* predictions, bool verbose = false);
RF_metrics score(const cumlHandle& user_handle,
                 const rfClassifier<double>* rf_classifier, const double* input,
                 const int* ref_labels, int n_rows, int n_cols,
                 int* predictions, bool verbose = false);
void predictAllDTs(const cumlHandle& user_handle,
                   const rfClassifier<float>* rf_classifier, const float* input,
                   int n_rows, int n_cols, int* predictions,
                   bool verbose = false);
void predictAllDTs(const cumlHandle& user_handle,
                   const rfClassifier<double>* rf_classifier,
                   const double* input, int n_rows, int n_cols,
                   int* predictions, bool verbose = false);


RF_params set_rf_class_obj(int max_depth, int max_leaves, float max_features,
                           int n_bins, int split_algo, int min_rows_per_node,
                           bool bootstrap_features, bool bootstrap, int n_trees,
                           float rows_sample, CRITERION split_criterion,
                           bool quantile_per_tree);

// ----------------------------- Regression ----------------------------------- //

void fit(const cumlHandle& user_handle, rfRegressor<float>* rf_regressor,
         float* input, int n_rows, int n_cols, float* labels);
void fit(const cumlHandle& user_handle, rfRegressor<double>* rf_regressor,
         double* input, int n_rows, int n_cols, double* labels);

void predict(const cumlHandle& user_handle,
             const rfRegressor<float>* rf_regressor, const float* input,
             int n_rows, int n_cols, float* predictions, bool verbose = false);
void predict(const cumlHandle& user_handle,
             const rfRegressor<double>* rf_regressor, const double* input,
             int n_rows, int n_cols, double* predictions, bool verbose = false);

RF_metrics score(const cumlHandle& user_handle,
                 const rfRegressor<float>* rf_regressor,
                 const float* input, const float* ref_labels,
                 int n_rows, int n_cols, float* predictions,
                 bool verbose = false);
RF_metrics score(const cumlHandle& user_handle,
                 const rfRegressor<double>* rf_regressor,
                 const double* input, const double* ref_labels,
                 int n_rows, int n_cols, double* predictions,
                 bool verbose = false);
};  // namespace ML
