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

  num_pred_bboxes_ = bottom[0]->num();
  CHECK_GT(num_pred_bboxes_, 0) << "predicted bboxes must be > 0";

  vector<int> top_shape(1, 1);
  top[0]->Reshape(top_shape);
}

template<typename Dtype>
void HungarianLossLayer<Dtype>::PrepareForBBoxes(
    const Blob<Dtype> *gt_bbox_blob) {
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

  total_bboxes_ = batch_size_ * num_pred_bboxes_;
}

template<typename Dtype>
Dtype HungarianLossLayer<Dtype>::ForwardBatch(
    const vector<Blob<Dtype> *> &bottom, int batch_ind) {
  const Dtype *bbox_pred = bottom[0]->cpu_data();
  const Dtype *bbox_gt = bottom[1]->cpu_data();
  const Dtype *labels = bottom[2]->cpu_data();
  const Dtype *prob = prob_.cpu_data();
  const Dtype *conf = confidence_.cpu_data();

  // Compute cost matrix.
  // const Dtype *bbox_pred_batch = bbox_pred;
  const Dtype *bbox_gt_batch = bbox_gt +
      bottom[1]->offset(start_idx_vec_[batch_ind]);
  const Dtype *labels_batch = labels +
      bottom[2]->offset(start_idx_vec_[batch_ind]);

  int num_gt_bboxes = num_gt_bboxes_vec_[batch_ind];

  vector<double> cost(num_pred_bboxes_ * num_gt_bboxes, 0.);
  for (int i = 0; i < num_pred_bboxes_; ++i) {
    for (int j = 0; j < num_gt_bboxes; ++j) {
      const int idx = i * num_gt_bboxes + j;
      // location loss
      for (int c = 0; c < bottom[0]->height(); ++c) {
        Dtype pred_value = bbox_pred[bottom[0]->offset(i, batch_ind, c)];
        Dtype gt_value = bbox_gt_batch[bottom[1]->offset(j, c + 1)];
        cost[idx] += fabs(pred_value - gt_value);
      }
      // classification loss
      int label = static_cast<int>(labels_batch[bottom[2]->offset(j)]);
      CHECK_NEAR(label, labels_batch[bottom[2]->offset(j)], 0.01);
      cost[idx] -= cls_weight_ *
          log(max(prob[prob_.offset(i, batch_ind, label)], Dtype(FLT_MIN)));
    }
  }

  // Solve the assignment problem.
  hungarian_problem_t p;
  double **m = array_to_matrix(cost.data(), num_pred_bboxes_, num_gt_bboxes);
  int matrix_size = hungarian_init(&p, m, num_pred_bboxes_, num_gt_bboxes,
                                   HUNGARIAN_MODE_MINIMIZE_COST);
  hungarian_solve(&p);
  for (int i = 0; i < num_pred_bboxes_; ++i) {
    for (int j = 0; j < matrix_size; ++j) {
      if (p.assignment[i][j] == HUNGARIAN_ASSIGNED) {
        assignment_[batch_ind].push_back(j);
      }
    }
  }
  CHECK_EQ(assignment_[batch_ind].size(), num_pred_bboxes_);
  hungarian_free(&p);
  for (int i = 0; i < num_pred_bboxes_; ++i) {
    free(m[i]);
  }
  free(m);

  // Compute loss.
  Dtype loss = 0.;
  for (int i = 0; i < num_pred_bboxes_; ++i) {
    int match_gt_ind = assignment_[batch_ind][i];
    const int idx = i * num_gt_bboxes + match_gt_ind;
    if (match_gt_ind < num_gt_bboxes) {
      loss += cost[idx];
    }
    else {
      loss -= cls_weight_ *
          log(max(prob[prob_.offset(i, batch_ind, 0)], Dtype(FLT_MIN)));
    }
    // Objectness loss
    if (i < num_gt_bboxes) {
      loss -= obj_weight_ *
          log(max(conf[confidence_.offset(i, batch_ind, 1)], Dtype(FLT_MIN)));
    }
    else {
      loss -= obj_weight_ *
          log(max(conf[confidence_.offset(i, batch_ind, 0)], Dtype(FLT_MIN)));
    }
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

  PrepareForBBoxes(bottom[1]);

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

  int num_gt_bboxes = num_gt_bboxes_vec_[batch_ind];

  for (int i = 0; i < num_pred_bboxes_; ++i) {
    const int match_gt_ind = assignment_[batch_ind][i];
    if (match_gt_ind >= num_gt_bboxes) {
      cls_score_diff[bottom[4]->offset(i, batch_ind, 0)] -= 1;
    }
    else {
      int label = labels[bottom[2]->offset(match_gt_ind)];
      cls_score_diff[bottom[4]->offset(i, batch_ind, label)] -= 1;
      for (int c = 0; c < bottom[0]->height(); ++c) {
        const Dtype pred_value = bbox_pred[bottom[0]->offset(i, batch_ind, c)];
        const Dtype gt_value = bbox_gt[bottom[1]->offset(match_gt_ind, c + 1)];
        bbox_pred_diff[bottom[0]->offset(i, batch_ind, c)] =
            pred_value > gt_value ? Dtype(1) : Dtype(-1);
      }
    }

    if (i < num_gt_bboxes) {
      confidence_diff[bottom[3]->offset(i, batch_ind, 1)] -= 1;
    }
    else {
      confidence_diff[bottom[3]->offset(i, batch_ind, 0)] -= 1;
    }
  }  // for (i = 0; i < num_pred_bboxes_truth; ++i)
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
  const Dtype *conf = confidence_.cpu_data();
  Dtype *bbox_pred_diff = bottom[0]->mutable_cpu_diff();
  Dtype *cls_score_diff = bottom[4]->mutable_cpu_diff();
  Dtype *confidence_diff = bottom[3]->mutable_cpu_diff();
  caffe_set(bottom[0]->count(), Dtype(0), bbox_pred_diff);
  caffe_copy(prob_.count(), prob, cls_score_diff);
  caffe_copy(confidence_.count(), conf, confidence_diff);

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
