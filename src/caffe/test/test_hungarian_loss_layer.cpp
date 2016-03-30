#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/hungarian_loss_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

template <typename TypeParam>
class HungarianLossLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;
  
 protected:
  HungarianLossLayerTest() {
    top_loss_ = new Blob<Dtype>();
    top_vec_.push_back(top_loss_);
  }
  virtual ~HungarianLossLayerTest() {
    delete bbox_pred_;
    delete bbox_gt_;
    delete labels_;
    delete confidence_;
    delete cls_scores_;
    delete top_loss_;
  }
  Blob<Dtype> *bbox_pred_;
  Blob<Dtype> *bbox_gt_;
  Blob<Dtype> *labels_;
  Blob<Dtype> *confidence_;
  Blob<Dtype> *cls_scores_;
  Blob<Dtype> *top_loss_;
  vector<Blob<Dtype>*> bottom_vec_;
  vector<Blob<Dtype>*> top_vec_;
};

TYPED_TEST_CASE(HungarianLossLayerTest, TestDtypesAndDevices);

/*TYPED_TEST(HungarianLossLayerTest, TestForward) {
  LayerParameter layer_param;
  HungarianLossLayer<Dtype> layer(layer_param);
  layer.SetUp(bottom_vec_, top_vec_);
  const vector<int> &assignment = layer.assignment();
  EXPECT_EQ(assignment[0], 0);
  EXPECT_EQ(assignment[1], 2);
  EXPECT_EQ(assignment[2], 1);
}*/

TYPED_TEST(HungarianLossLayerTest, TestGradientPredEnough) {
  typedef typename TypeParam::Dtype Dtype;
  std::vector<int> bbox_gt_shape(2);
  bbox_gt_shape[0] = 2;
  bbox_gt_shape[1] = 4;
  this->bbox_gt_ = new Blob<Dtype>(bbox_gt_shape);
  Dtype *bbox_gt_data = this->bbox_gt_->mutable_cpu_data();
  Dtype bbox_gt_data_array[8] = { 0.2364, 0.1604, 0.3853, 0.3797,
      0.6856, 0.6551, 0.8877 };
  for (int i = 0; i < 8; ++i) {
    bbox_gt_data[i] = bbox_gt_data_array[i];
  }
  std::vector<int> bbox_pred_shape(2);
  bbox_pred_shape[0] = 3;
  bbox_pred_shape[1] = 4;
  this->bbox_pred_ = new Blob<Dtype>(bbox_pred_shape);
  Dtype *bbox_pred_data = this->bbox_pred_->mutable_cpu_data();
  Dtype bbox_pred_data_array[12] = { 0.1678, 0.2674, 0.3097, 0.4759,
      0.2861, 0.0989, 0.4161, 0.3556, 0.6265, 0.7246, 0.7565, 0.9519 };
  for (int i = 0; i < 12; ++i) {
    bbox_pred_data[i] = bbox_pred_data_array[i];
  }
  std::vector<int> labels_shape(1, 2);
  this->labels_ = new Blob<Dtype>(labels_shape);
  Dtype *labels_data = this->labels_->mutable_cpu_data();
  labels_data[0] = 3;
  labels_data[1] = 15;
  std::vector<int> confidence_shape(2);
  confidence_shape[0] = 3;
  confidence_shape[1] = 2;
  this->confidence_ = new Blob<Dtype>(confidence_shape);
  Dtype *confidence_data = this->confidence_->mutable_cpu_data();
  Dtype confidence_data_array[6] = { 0.1, 0.4, 0.15, 0.35, 0.2, 0.3 };
  for (int i = 0; i < 6; ++i) {
    confidence_data[i] = confidence_data_array[i];
  }
  std::vector<int> cls_scores_shape(2);
  cls_scores_shape[0] = 3;
  cls_scores_shape[1] = 21;
  this->cls_scores_ = new Blob<Dtype>(cls_scores_shape);
  Dtype *cls_scores_data = this->cls_scores_->mutable_cpu_data();
  int cls[3] = { 3, 3, 15 };
  for (int i = 0; i < this->cls_scores_->num(); ++i) {
    for (int j = 0; j < this->cls_scores_->channels(); ++j) {
      int idx = i*this->cls_scores_->channels() + j;
      if (j == cls[i]) {
        cls_scores_data[idx] = 0.8;
      }
      else {
        cls_scores_data[idx] = 0.2 / (this->cls_scores_->channels()-1);
      }
    }
  } 
  this->bottom_vec_.push_back(this->bbox_pred_);
  this->bottom_vec_.push_back(this->bbox_gt_);
  this->bottom_vec_.push_back(this->labels_);
  this->bottom_vec_.push_back(this->confidence_);
  this->bottom_vec_.push_back(this->cls_scores_);
  LayerParameter layer_param;
  layer_param.add_loss_weight(3);
  layer_param.mutable_hungarian_loss_param()->set_obj_weight(0.1);
  layer_param.mutable_hungarian_loss_param()->set_cls_weight(0.1);
  HungarianLossLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-2, 1701);
  checker.CheckGradientExhaustive(&layer, this->bottom_vec_,
      this->top_vec_, 0);
  checker.CheckGradientExhaustive(&layer, this->bottom_vec_,
      this->top_vec_, 3);
  checker.CheckGradientExhaustive(&layer, this->bottom_vec_,
      this->top_vec_, 4);
}

TYPED_TEST(HungarianLossLayerTest, TestGradientPredNotEnough) {
  typedef typename TypeParam::Dtype Dtype;
  std::vector<int> bbox_pred_shape(2);
  bbox_pred_shape[0] = 2;
  bbox_pred_shape[1] = 4;
  this->bbox_pred_ = new Blob<Dtype>(bbox_pred_shape);
  Dtype *bbox_pred_data = this->bbox_pred_->mutable_cpu_data();
  Dtype bbox_pred_data_array[8] = { 0.2364, 0.1604, 0.3853, 0.3797,
        0.6856, 0.6551, 0.8877 };
  for (int i = 0; i < 8; ++i) {
    bbox_pred_data[i] = bbox_pred_data_array[i];
  }
  std::vector<int> bbox_gt_shape(2);
  bbox_gt_shape[0] = 3;
  bbox_gt_shape[1] = 4;
  this->bbox_gt_ = new Blob<Dtype>(bbox_gt_shape);
  Dtype *bbox_gt_data = this->bbox_gt_->mutable_cpu_data();
  Dtype bbox_gt_data_array[12] = { 0.1678, 0.2674, 0.3097, 0.4759,
      0.2861, 0.0989, 0.4161, 0.3556, 0.6265, 0.7246, 0.7565, 0.9519 };
  for (int i = 0; i < 12; ++i) {
    bbox_gt_data[i] = bbox_gt_data_array[i];
  }
  std::vector<int> labels_shape(1, 3);
  this->labels_ = new Blob<Dtype>(labels_shape);
  Dtype *labels_data = this->labels_->mutable_cpu_data();
  labels_data[0] = 3;
  labels_data[1] = 15;
  labels_data[2] = 5;
  std::vector<int> confidence_shape(2);
  confidence_shape[0] = 2;
  confidence_shape[1] = 2;
  this->confidence_ = new Blob<Dtype>(confidence_shape);
  Dtype *confidence_data = this->confidence_->mutable_cpu_data();
  Dtype confidence_data_array[6] = { 0.1, 0.4, 0.15, 0.35 };
  for (int i = 0; i < 4; ++i) {
    confidence_data[i] = confidence_data_array[i];
  }
  std::vector<int> cls_scores_shape(2);
  cls_scores_shape[0] = 2;
  cls_scores_shape[1] = 21;
  this->cls_scores_ = new Blob<Dtype>(cls_scores_shape);
  Dtype *cls_scores_data = this->cls_scores_->mutable_cpu_data();
  int cls[2] = { 3, 15 };
  for (int i = 0; i < this->cls_scores_->num(); ++i) {
    for (int j = 0; j < this->cls_scores_->channels(); ++j) {
      int idx = i*this->cls_scores_->channels() + j;
      if (j == cls[i]) {
        cls_scores_data[idx] = 0.8;
      }
      else {
        cls_scores_data[idx] = 0.2 / (this->cls_scores_->channels()-1);
      }
    }
  } 
  this->bottom_vec_.push_back(this->bbox_pred_);
  this->bottom_vec_.push_back(this->bbox_gt_);
  this->bottom_vec_.push_back(this->labels_);
  this->bottom_vec_.push_back(this->confidence_);
  this->bottom_vec_.push_back(this->cls_scores_);
  LayerParameter layer_param;
  layer_param.add_loss_weight(3);
  HungarianLossLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-2, 1701);
  checker.CheckGradientExhaustive(&layer, this->bottom_vec_,
      this->top_vec_, 0);
  checker.CheckGradientExhaustive(&layer, this->bottom_vec_,
      this->top_vec_, 3);
  checker.CheckGradientExhaustive(&layer, this->bottom_vec_,
      this->top_vec_, 4);
}

}  // namespace caffe