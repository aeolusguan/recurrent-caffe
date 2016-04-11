//
// Created by aeolus on 16-3-16.
//
#include <cfloat>
#include <numeric>

#include "caffe/hungarian_loss_layer.hpp"
#include "caffe/util/hungarian.hpp"
#include "caffe/layer_factory.hpp"

namespace caffe {

using std::max;
using std::fabs;
using std::fabs;

template<typename Dtype>
void HungarianLossLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype> *> &bottom,
                                           const vector<Blob<Dtype> *> &top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  // Create softmax layer for object confidence.
  LayerParameter softmax_obj_param;
  softmax_obj_param.set_type("Softmax");
  softmax_obj_param.add_bottom("confidence");
  softmax_obj_param.add_top("normalized_confidence");
  softmax_obj_param.mutable_softmax_param()->set_axis(2);
  softmax_obj_layer_ = LayerRegistry<Dtype>::CreateLayer(softmax_obj_param);
  softmax_obj_bottom_vec_.clear();
  softmax_obj_bottom_vec_.push_back(bottom[3]);
  softmax_obj_top_vec_.clear();
  softmax_obj_top_vec_.push_back(&confidence_);
  softmax_obj_layer_->SetUp(softmax_obj_bottom_vec_, softmax_obj_top_vec_);
  // Create softmax layer for classification scores.
  LayerParameter softmax_cls_param;
  softmax_cls_param.set_type("Softmax");
  softmax_cls_param.add_bottom("cls_score");
  softmax_cls_param.add_top("normalized_cls_score");
  softmax_cls_param.mutable_softmax_param()->set_axis(2);
  softmax_cls_layer_ = LayerRegistry<Dtype>::CreateLayer(softmax_cls_param);
  softmax_cls_bottom_vec_.clear();
  softmax_cls_bottom_vec_.push_back(bottom[4]);
  softmax_cls_top_vec_.clear();
  softmax_cls_top_vec_.push_back(&prob_);
  softmax_cls_layer_->SetUp(softmax_cls_bottom_vec_, softmax_cls_top_vec_);

  HungarianLossParameter hungarian_loss_param = this->layer_param_.
      hungarian_loss_param();
  CHECK_GE(hungarian_loss_param.obj_weight(), 0.f)
    << "objectness weight must be >= 0";
  CHECK_GE(hungarian_loss_param.cls_weight(), 0.f)
    << "classification weight must be >= 0";
  obj_weight_ = hungarian_loss_param.obj_weight();
  cls_weight_ = hungarian_loss_param.cls_weight();

  assignment_.reserve(bottom[0]->num());
  min_objectness_.reserve(bottom[0]->num());
  min_idx_.reserve(bottom[0]->num());
}

template<typename Dtype>
void HungarianLossLayer<Dtype>::Reshape(const vector<Blob<Dtype> *> &bottom,
                                        const vector<Blob<Dtype> *> &top) {
  // LossLayer<Dtype>::Reshape(bottom, top);
  softmax_obj_layer_->Reshape(softmax_obj_bottom_vec_, softmax_obj_top_vec_);
  softmax_cls_layer_->Reshape(softmax_cls_bottom_vec_, softmax_cls_top_vec_);

  CHECK_EQ(bottom[0]->height(), 4)
    << "height of predicted bounding box blob must be 4";
  CHECK_EQ(bottom[1]->channels(), 5)
    << "ground truth bounding box blob must have 5 channels";
  CHECK_EQ(bottom[3]->height(), 2)
    << "height of objectness confidence must be 2";
  CHECK_EQ(bottom[4]->height(), 21)
    << "height of classification score must be 21";
  batch_size_ = bottom[0]->channels();
  CHECK_GT(batch_size_, 0) << "batch size must be > 0";

  vector<int> top_shape(1, 1);
  top[0]->Reshape(top_shape);
}

template<typename Dtype>
void HungarianLossLayer<Dtype>::PrepareForBBoxes(
    const Blob<Dtype> *gt_bbox_blob, int num_pred_bboxes) {
  // Prepare for the number of ground truth bboxes
  // of each batch and the start index in the input
  // ground truth bbox bottom for each batch.
  int total_bboxes = gt_bbox_blob->num();
  const Dtype *gt_bbox_blob_data = gt_bbox_blob->cpu_data();
  num_gt_bboxes_vec_.clear();
  num_gt_bboxes_vec_.reserve(batch_size_);
  start_idx_vec_.clear();
  start_idx_vec_.reserve(batch_size_);
  // The start index for the first batch is 0.
  start_idx_vec_.push_back(0);
  // The batch index for the first batch is 0,
  // which is matched with the input blob.
  int batch_ind = 0;
  int count = 1;
  gt_bbox_blob_data += gt_bbox_blob->offset(1);
  for (int i = 1; i < total_bboxes; ++i) {
    if (gt_bbox_blob_data[0] != batch_ind) {
      num_gt_bboxes_vec_.push_back(count);
      start_idx_vec_.push_back(i);
      batch_ind++;
      count = 1;
    }
    else {
      count++;
    }
    gt_bbox_blob_data += gt_bbox_blob->offset(1);
  }
  num_gt_bboxes_vec_.push_back(count);
  CHECK_EQ(num_gt_bboxes_vec_.size(), batch_size_);
  CHECK_EQ(start_idx_vec_.size(), batch_size_);

  // Compute the number of predicted bboxes for assignment for
  // each batch.
  num_pred_bboxes_.clear();
  num_pred_bboxes_.resize(batch_size_);
  num_pred_bboxes_truth_.clear();
  num_pred_bboxes_truth_.resize(batch_size_);
  for (int i = 0; i < batch_size_; ++i) {
    int num_gt_bboxes = num_gt_bboxes_vec_[i];
    num_pred_bboxes_[i] = num_pred_bboxes >= num_gt_bboxes + 1 ?
                          num_gt_bboxes + 1 : num_gt_bboxes;
    num_pred_bboxes_truth_[i] = num_pred_bboxes >= num_pred_bboxes_[i] ?
                                num_pred_bboxes_[i] : num_pred_bboxes;
  }
  total_bboxes_ = std::accumulate(num_pred_bboxes_truth_.begin(),
                                  num_pred_bboxes_truth_.end(),
                                  0);
}

template<typename Dtype>
void HungarianLossLayer<Dtype>::PrepareForConf() {
  const Dtype *confidence = confidence_.cpu_data();

  // Compute min_objectness_ and min_idx_.
  ClearVec(min_objectness_);
  min_objectness_.resize(batch_size_);
  ClearVec(min_idx_);
  min_idx_.resize(batch_size_);

  for (int i = 0; i < batch_size_; ++i) {
    const Dtype *confidence_batch =
        confidence + confidence_.offset(0, i, 1);
    min_objectness_[i].push_back(confidence_batch[0]);
    min_idx_[i].push_back(0);
    confidence_batch += confidence_.offset(1);
    for (int k = 1; k < num_pred_bboxes_truth_[i]; ++k) {
      if (confidence_batch[0] < min_objectness_[i][k - 1]) {
        min_objectness_[i].push_back(confidence_batch[0]);
        min_idx_[i].push_back(k);
      }
      else {
        min_objectness_[i].push_back(min_objectness_[i][k - 1]);
        min_idx_[i].push_back(min_idx_[i][k - 1]);
      }
      confidence_batch += confidence_.offset(1);
    }
  }
}

template<typename Dtype>
Dtype HungarianLossLayer<Dtype>::ForwardBatch(
    const vector<Blob<Dtype> *> &bottom, int batch_ind) {
  const Dtype *bbox_pred = bottom[0]->cpu_data();
  const Dtype *bbox_gt = bottom[1]->cpu_data();
  const Dtype *labels = bottom[2]->cpu_data();
  const Dtype *prob = prob_.cpu_data();

  // Compute cost matrix.
  // const Dtype *bbox_pred_batch = bbox_pred;
  const Dtype *bbox_gt_batch = bbox_gt +
      bottom[1]->offset(start_idx_vec_[batch_ind]);
  const Dtype *labels_batch = labels +
      bottom[2]->offset(start_idx_vec_[batch_ind]);
  int num_pred_bboxes = num_pred_bboxes_[batch_ind];
  int num_pred_bboxes_truth = num_pred_bboxes_truth_[batch_ind];
  int num_gt_bboxes = num_gt_bboxes_vec_[batch_ind];
  vector<float> cost(num_pred_bboxes * num_pred_bboxes, 0.f);
  for (int i = 0; i < num_pred_bboxes_truth; ++i) {
    // i: index of predicted bbox
    for (int j = 0; j < num_gt_bboxes; ++j) {
      // j: index of ground truth bbox
      const int idx = i * num_pred_bboxes + j;
      // location loss (L1 loss)
      for (int c = 0; c < bottom[0]->height(); ++c) {
        const Dtype pred_value = bbox_pred[bottom[0]->offset(i, batch_ind, c)];
        const Dtype gt_value = bbox_gt_batch[bottom[1]->offset(j, c + 1)];
        cost[idx] += fabs(pred_value - gt_value);
      }
      // objectness loss
      cost[idx] -= obj_weight_ *
          log(max(min_objectness_[batch_ind][i], Dtype(FLT_MIN)));
      // classification loss
      int label = static_cast<int>(labels_batch[bottom[2]->offset(j)]);
      CHECK_NEAR(label, labels_batch[bottom[2]->offset(j)], 0.01);
      cost[idx] -= cls_weight_ *
          log(max(prob[prob_.offset(i, batch_ind, label)], Dtype(FLT_MIN)));
    }
    if (num_pred_bboxes > num_gt_bboxes) {
      // cost for predicted bbox i not assigned to any ground truth bbox
      cost[(i + 1) * num_pred_bboxes - 1] -= static_cast<float>(
          obj_weight_ * log(max(1 - min_objectness_[batch_ind][i],
                                Dtype(FLT_MIN)))
              + cls_weight_ * log(max(prob[prob_.offset(i, batch_ind, 0)],
                                      Dtype(FLT_MIN))));
    }
  }

  // Solve the assignment problem.
  hungarian_problem_t p;
  const int scale = 1000;
  vector<int> cost_int(cost.size());
  for (int i = 0; i < cost.size(); ++i) {
    cost_int[i] = static_cast<int>(cost[i] * scale);
  }
  int **m = array_to_matrix(cost_int.data(), num_pred_bboxes, num_pred_bboxes);
  hungarian_init(&p, m, num_pred_bboxes, num_pred_bboxes,
                 HUNGARIAN_MODE_MINIMIZE_COST);
  hungarian_solve(&p);
  for (int i = 0; i < num_pred_bboxes; ++i) {
    for (int j = 0; j < num_pred_bboxes; ++j) {
      if (p.assignment[i][j] == HUNGARIAN_ASSIGNED) {
        assignment_[batch_ind].push_back(j);
      }
    }
  }
  CHECK_EQ(assignment_[batch_ind].size(), num_pred_bboxes);
  hungarian_free(&p);
  for (int i = 0; i < num_pred_bboxes; ++i) {
    free(m[i]);
  }
  free(m);

  // Compute loss.
  Dtype loss = 0.;
  for (int i = 0; i < num_pred_bboxes_truth; ++i) {
    const int idx = i * num_pred_bboxes + assignment_[batch_ind][i];
    loss += cost[idx];
  }

  return loss;
}

template<typename Dtype>
void HungarianLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype> *> &bottom,
                                            const vector<Blob<Dtype> *> &top) {
  softmax_obj_layer_->Forward(softmax_obj_bottom_vec_, softmax_obj_top_vec_);
  softmax_cls_layer_->Forward(softmax_cls_bottom_vec_, softmax_cls_top_vec_);

  ClearVec(assignment_);
  assignment_.resize(batch_size_);

  PrepareForBBoxes(bottom[1], bottom[0]->num());
  PrepareForConf();

  Dtype loss = 0;
  for (int i = 0; i < batch_size_; ++i) {
    loss += ForwardBatch(bottom, i);
  }
  loss /= total_bboxes_;
  top[0]->mutable_cpu_data()[0] = loss;
}

template<typename Dtype>
void HungarianLossLayer<Dtype>::BackwardBatch(
    const vector<Blob<Dtype> *> &bottom, int batch_ind) {
  const Dtype *bbox_pred = bottom[0]->cpu_data();
  const Dtype *bbox_gt = bottom[1]->cpu_data();
  bbox_gt += bottom[1]->offset(start_idx_vec_[batch_ind]);
  const Dtype *labels = bottom[2]->cpu_data();
  labels += bottom[2]->offset(start_idx_vec_[batch_ind]);

  Dtype *bbox_pred_diff = bottom[0]->mutable_cpu_diff();
  Dtype *confidence_diff = bottom[3]->mutable_cpu_diff();
  Dtype *cls_score_diff = bottom[4]->mutable_cpu_diff();

  int num_pred_bboxes_truth = num_pred_bboxes_truth_[batch_ind];
  int num_gt_bboxes = num_gt_bboxes_vec_[batch_ind];

  for (int i = 0; i < num_pred_bboxes_truth; ++i) {
    // i: index for predicted bbox
    int min_idx = min_idx_[batch_ind][i];
    if (assignment_[batch_ind][i] == num_gt_bboxes) {
      // not assigned to any ground truth bbox.
      // Backward objectness loss.
      confidence_diff[bottom[3]->offset(min_idx, batch_ind, 0)] +=
          -min_objectness_[batch_ind][i];
      confidence_diff[bottom[3]->offset(min_idx, batch_ind, 1)] +=
          min_objectness_[batch_ind][i];
      // Backward classification loss.
      cls_score_diff[bottom[4]->offset(i, batch_ind, 0)] -= 1;
    }
    else {
      // assigned to some ground truth bbox

      // Backward objectness loss.
      confidence_diff[bottom[3]->offset(min_idx, batch_ind, 0)] +=
          1 - min_objectness_[batch_ind][i];
      confidence_diff[bottom[3]->offset(min_idx, batch_ind, 1)] +=
          min_objectness_[batch_ind][i] - 1;
      // Backward classification loss.
      int label = labels[bottom[2]->offset(assignment_[batch_ind][i])];
      cls_score_diff[bottom[4]->offset(i, batch_ind, label)] -= 1;
      // Backward location loss.
      for (int c = 0; c < bottom[0]->height(); ++c) {
        const Dtype pred_value = bbox_pred[bottom[0]->offset(i, batch_ind, c)];
        const Dtype gt_value =
            bbox_gt[bottom[1]->offset(assignment_[batch_ind][i], c + 1)];
        bbox_pred_diff[bottom[0]->offset(i, batch_ind, c)] =
            pred_value > gt_value ? Dtype(1) : Dtype(-1);
      }
    }
  }  // for (i = 0; i < num_pred_bboxes_truth; ++i)

  for (int i = num_pred_bboxes_truth; i < bottom[0]->num(); ++i) {
    for (int c = 0; c < bottom[4]->height(); ++c) {
      cls_score_diff[bottom[4]->offset(i, batch_ind, c)] = Dtype(0);
    }
  }

}

template<typename Dtype>
void HungarianLossLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype> *> &top,
    const vector<bool> &propagate_down,
    const vector<Blob<Dtype> *> &bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
        << "Layer can not backpropagate ground truth bbox input";
  }
  if (propagate_down[2]) {
    LOG(FATAL) << this->type()
        << "Layer can not backpropagate label input";
  }

  const Dtype *prob = prob_.cpu_data();
  Dtype *bbox_pred_diff = bottom[0]->mutable_cpu_diff();
  Dtype *cls_score_diff = bottom[4]->mutable_cpu_diff();
  Dtype *confidence_diff = bottom[3]->mutable_cpu_diff();
  caffe_set(bottom[0]->count(), Dtype(0), bbox_pred_diff);
  caffe_copy(prob_.count(), prob, cls_score_diff);
  caffe_set(bottom[3]->count(), Dtype(0), confidence_diff);

  for (int i = 0; i < batch_size_; ++i) {
    BackwardBatch(bottom, i);
  }

  // Scale gradient
  Dtype *top_diff = top[0]->mutable_cpu_diff();
  Dtype loss_weight = top_diff[0];
  loss_weight /= total_bboxes_;
  caffe_scal(confidence_.count(), loss_weight * obj_weight_, confidence_diff);
  caffe_scal(prob_.count(), loss_weight * cls_weight_, cls_score_diff);
  caffe_scal(bottom[0]->count(), loss_weight, bbox_pred_diff);
}

#ifdef CPU_ONLY
STUB_GPU(HungarianLossLayer)
#endif

INSTANTIATE_CLASS(HungarianLossLayer);
REGISTER_LAYER_CLASS(HungarianLoss);

}  // namespace caffe
