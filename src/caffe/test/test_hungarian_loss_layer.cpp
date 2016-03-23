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
    std::vector<int> bbox_gt_shape(2);
    bbox_gt_shape[0] = 2;
    bbox_gt_shape[1] = 4;
    bbox_gt_ = new Blob<Dtype>(bbox_gt_shape);
    Dtype *bbox_gt_data = bbox_gt_->mutable_cpu_data();
    Dtype bbox_gt_data_array[8] = { 0.2364, 0.1604, 0.3853, 0.3797,
        0.6856, 0.6551, 0.8877 };
    for (int i = 0; i < 8; ++i) {
      bbox_gt_data[i] = bbox_gt_data_array[i];
    }
    std::vector<int> bbox_pred_shape(2);
    bbox_pred_shape[0] = 3;
    bbox_pred_shape[1] = 4;
    bbox_pred_ = new Blob<Dtype>(bbox_pred_shape);
    Dtype *bbox_pred_data = bbox_pred_->mutable_cpu_data();
    Dtype bbox_pred_data_array[12] = { 0.1678, 0.2674, 0.3097, 0.4759,
        0.2861, 0.0989, 0.4161, 0.3556, 0.6265, 0.7246, 0.7565, 0.9519 };
    for (int i = 0; i < 12; ++i) {
      bbox_pred_data[i] = bbox_pred_data_array[i];
    }
    std::vector<int> labels_shape(1, 2);
    labels_ = new Blob<Dtype>(labels_shape);
    Dtype *labels_data = labels_->mutable_cpu_data();
    labels_data[0] = 3;
    labels_data[1] = 15;
    std::vector<int> confidence_shape(2);
    confidence_shape[0] = 3;
    confidence_shape[1] = 2;
    confidence_ = new Blob<Dtype>(confidence_shape);
    Dtype *confidence_data = confidence_->mutable_cpu_data();
    Dtype confidence_data_array[6] = { 0.1, 0.4, 0.15, 0.35, 0.2, 0.3 };
    for (int i = 0; i < 6; ++i) {
      confidence_data[i] = confidence_data_array[i];
    }
    std::vector<int> cls_scores_shape(2);
    cls_scores_shape[0] = 3;
    cls_scores_shape[1] = 21;
    cls_scores_ = new Blob<Dtype>(cls_scores_shape);
    Dtype *cls_scores_data = cls_scores_->mutable_cpu_data();
    int cls[3] = { 3, 3, 15 };
    for (int i = 0; i < cls_scores_->num(); ++i) {
      for (int j = 0; j < cls_scores_->channels(); ++j) {
        int idx = i*cls_scores_->channels() + j;
        if (j == cls[i]) {
          cls_scores_data[idx] = 0.8;
        }
        else {
          cls_scores_data[idx] = 0.2 / (cls_scores_->channels()-1);
        }
      }
    } 
    top_loss_ = new Blob<Dtype>();
    bottom_vec_.push_back(bbox_pred_);
    bottom_vec_.push_back(bbox_gt_);
    bottom_vec_.push_back(labels_);
    bottom_vec_.push_back(confidence_);
    bottom_vec_.push_back(cls_scores_);
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

TYPED_TEST(HungarianLossLayerTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  HungarianLossLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-2, 1701);
  checker.CheckGradientExhaustive(&layer, this->bottom_vec_,
      this->top_vec_, 0);
}

}  // namespace caffe