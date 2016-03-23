//
// Created by aeolus on 16-3-16.
//
#include <cfloat>

#include "caffe/hungarian_loss_layer.hpp"
#include "caffe/util/hungarian.hpp"
#include "caffe/layer_factory.hpp"

namespace caffe {

template <typename Dtype>
void HungarianLossLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*> &bottom,
                                           const vector<Blob<Dtype>*> &top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  // Create softmax layer for object confidence.
  LayerParameter softmax_obj_param;
  softmax_obj_param.set_type("Softmax");
  softmax_obj_param.add_bottom("confidence");
  softmax_obj_param.add_top("normalized_confidence");
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
}

template <typename Dtype>
void HungarianLossLayer<Dtype>::Reshape(const vector<Blob<Dtype>*> &bottom,
                                        const vector<Blob<Dtype>*> &top) {
  // LossLayer<Dtype>::Reshape(bottom, top);
  softmax_obj_layer_->Reshape(softmax_obj_bottom_vec_, softmax_obj_top_vec_);
  softmax_cls_layer_->Reshape(softmax_cls_bottom_vec_, softmax_cls_top_vec_);
  CHECK_EQ(bottom[0]->num(), bottom[1]->num()+1)
      << "number of predicted bounding boxes must be that"
         " of ground truth plus 1";
  CHECK_EQ(bottom[0]->channels(), 4)
      << "predicted bounding box blob must have 4 channels";
  CHECK_EQ(bottom[1]->channels(), 4)
      << "ground truth bounding box blob must have 4 channels";
  CHECK_EQ(bottom[3]->channels(), 2)
      << "channels of objectness confidence must be 2";
  CHECK_EQ(bottom[4]->channels(), 21)
      << "channels of classification score must be 21";
  vector<int> top_shape(1, 1);
  top[0]->Reshape(top_shape);
  assignment_.reserve(bottom[0]->num());
  min_objectness_.reserve(bottom[0]->num());
  min_idx_.reserve(bottom[0]->num());
}

template <typename Dtype>
void HungarianLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*> &bottom,
                                            const vector<Blob<Dtype>*> &top) {
  softmax_obj_layer_->Forward(softmax_obj_bottom_vec_, softmax_obj_top_vec_);
  softmax_cls_layer_->Forward(softmax_cls_bottom_vec_, softmax_cls_top_vec_);
  const Dtype *bbox_pred = bottom[0]->cpu_data();
  const Dtype *bbox_gt = bottom[1]->cpu_data();
  const Dtype *labels = bottom[2]->cpu_data();
  const Dtype *confidence = confidence_.cpu_data();
  const Dtype *prob = prob_.cpu_data();

  const int num_pred_bboxes = bottom[0]->num();
  
  // Compute min_objectness_ and min_idx_.
  min_objectness_.clear();
  min_objectness_.push_back(confidence[confidence_.offset(0, 1)]);
  min_idx_.clear();
  min_idx_.push_back(0);
  for (int k = 1; k < num_pred_bboxes; ++k) {
    if (confidence[confidence_.offset(k, 1)] < min_objectness_[k-1]) {
      min_objectness_.push_back(confidence[confidence_.offset(k, 1)]);
      min_idx_.push_back(k);
    }
    else {
      min_objectness_.push_back(min_objectness_[k-1]);
      min_idx_.push_back(min_idx_[k-1]);
    }
  }
  
  // Compute cost matrix.
  vector<float> cost(num_pred_bboxes*num_pred_bboxes, 0.f);
  for (int i = 0; i < num_pred_bboxes; ++i) {  // i: index of predicted bbox
    for (int j = 0; j < num_pred_bboxes-1; ++j) {  // j: index of ground
                                                   // truth bbox
      const int idx = i*num_pred_bboxes + j;
      // location loss (L1 loss)
      for (int c = 0; c < bottom[0]->channels(); ++c) {
        const Dtype pred_value = bbox_pred[bottom[0]->offset(i, c)];
        const Dtype gt_value = bbox_gt[bottom[1]->offset(j, c)];
        cost[idx] += std::fabs(pred_value - gt_value);
      }
      // objectness loss
      cost[idx] -= obj_weight_*std::log(std::max(min_objectness_[i],
                                                 Dtype(FLT_MIN)));
      // classification loss
      int label = static_cast<int>(labels[bottom[2]->offset(j)]);
      CHECK_NEAR(label, labels[bottom[2]->offset(j)], 0.01);
      cost[idx] -= cls_weight_*std::log(std::max(prob[prob_.offset(i, label)],
                                                 Dtype(FLT_MIN)));
    }
    // cost for predicted bbox i not assigned to any ground truth bbox
    cost[(i+1)*num_pred_bboxes-1] -= obj_weight_*
        std::log(std::max(1-min_objectness_[i], Dtype(FLT_MIN))) + cls_weight_*
        std::log(std::max(prob[prob_.offset(i, 0)], Dtype(FLT_MIN)));
  }

  // Solve the assignment problem.
  assignment_.clear();
  hungarian_problem_t p;
  const int scale = 1000;
  vector<int> cost_int(cost.size());
  for (int i = 0; i < cost.size(); ++i) {
    cost_int[i] = static_cast<int>(cost[i]*scale);
  }
  int **m = array_to_matrix(cost_int.data(), num_pred_bboxes, num_pred_bboxes);
  hungarian_init(&p, m, num_pred_bboxes, num_pred_bboxes,
      HUNGARIAN_MODE_MINIMIZE_COST);
  hungarian_solve(&p);
  for (int i = 0; i < num_pred_bboxes; ++i) {
    for (int j = 0; j < num_pred_bboxes; ++j) {
      if (p.assignment[i][j] == HUNGARIAN_ASSIGNED) {
        assignment_.push_back(j);
      }
    }
  }
  CHECK_EQ(assignment_.size(), num_pred_bboxes);
  hungarian_free(&p);
  for (int i = 0; i < num_pred_bboxes; ++i) {
    free(m[i]);
  }
  free(m);

  // Compute loss.
  Dtype loss = 0.;
  for (int i = 0; i < num_pred_bboxes; ++i) {
    const int idx = i*num_pred_bboxes + assignment_[i];
    loss += cost[idx];
  }
  top[0]->mutable_cpu_data()[0] = loss;
}

template <typename Dtype>
void HungarianLossLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype>*> &top,
    const vector<bool> &propagate_down,
    const vector<Blob<Dtype>*> &bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << "Layer can not backpropagate ground truth bbox input";
  }
  if (propagate_down[2]) {
    LOG(FATAL) << this->type()
               << "Layer can not backpropagate label input"; 
  }
  const Dtype *bbox_pred = bottom[0]->cpu_data();
  const Dtype *bbox_gt = bottom[1]->cpu_data();
  const Dtype *prob = prob_.cpu_data();
  const Dtype *labels = bottom[2]->cpu_data();
  const Dtype *top_diff = top[0]->cpu_diff();

  Dtype *bbox_pred_diff = bottom[0]->mutable_cpu_diff();
  Dtype *confidence_diff = bottom[3]->mutable_cpu_diff();
  Dtype *cls_score_diff = bottom[4]->mutable_cpu_diff();
  caffe_set(bottom[0]->count(), Dtype(0), bbox_pred_diff);
  caffe_copy(prob_.count(), prob, cls_score_diff);
  // caffe_set(bottom[3]->count(), Dtype(0), objectness_diff);

  const int num_pred_bboxes = bottom[0]->num();
  for (int i = 0; i < num_pred_bboxes; ++i) {  // index for predicted 
                                               // ground truth
    int min_idx = min_idx_[i];
    if (assignment_[i] == num_pred_bboxes-1) {  // not assigned to any
                                                // ground truth bbox
      // Backward objectness loss.
      confidence_diff[bottom[3]->offset(min_idx, 0)] = -min_objectness_[i];
      confidence_diff[bottom[3]->offset(min_idx, 1)] = min_objectness_[i];
      // Backward classification loss.
      cls_score_diff[bottom[4]->offset(i, 0)] -= 1;
    }
    else {  // assigned to some ground truth bbox
      // Backward objectness loss.
      confidence_diff[bottom[3]->offset(min_idx, 0)] = 1-min_objectness_[i];
      confidence_diff[bottom[3]->offset(min_idx, 1)] = min_objectness_[i]-1;
      // Backward classification loss.
      int label = labels[bottom[2]->offset(assignment_[i])];
      cls_score_diff[bottom[4]->offset(i, label)] -= 1;
      // Backward location loss.
      for (int c = 0; c < bottom[0]->channels(); ++c) {
        const Dtype pred_value = bbox_pred[bottom[0]->offset(i, c)];
        const Dtype gt_value = bbox_gt[bottom[1]->offset(assignment_[i], c)];
        bbox_pred_diff[bottom[0]->offset(i, c)] = pred_value > gt_value ? 
            Dtype(1) : Dtype(-1); 
      }
    }
  }  // for (i = 0; i < num_pred_bboxes; ++i)

  // Scale gradient
  const Dtype loss_weight = top_diff[0];
  caffe_scal(confidence_.count(), loss_weight*obj_weight_, confidence_diff);
  caffe_scal(prob_.count(), loss_weight*cls_weight_, cls_score_diff);
  caffe_scal(bottom[0]->count(), loss_weight, bbox_pred_diff);
}

#ifdef CPU_ONLY
STUB_GPU(HungarianLossLayer)
#endif

INSTANTIATE_CLASS(HungarianLossLayer);
REGISTER_LAYER_CLASS(HungarianLoss);

}  // namespace caffe
