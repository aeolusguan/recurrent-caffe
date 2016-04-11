#include <vector>
#include <caffe/filler.hpp>

#include "gtest/gtest.h"

#include "caffe/hungarian_loss_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

template<typename TypeParam>
class HungarianLossLayerTest: public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  HungarianLossLayerTest() {
    top_loss_ = new Blob<Dtype>();
    top_vec_.push_back(top_loss_);

    vector<int> bbox_pred_shape(3);
    bbox_pred_shape[0] = 5;  // timestep: 1
    bbox_pred_shape[1] = 3;  // number of streams: 3
    bbox_pred_shape[2] = 4;  // coords of each bbox: 4
    bbox_pred_ = new Blob<Dtype>(bbox_pred_shape);
    FillerParameter fill_param;
    fill_param.set_min(0);
    fill_param.set_max(1);
    UniformFiller<Dtype> filler(fill_param);
    filler.Fill(bbox_pred_);

    vector<int> bbox_gt_shape(2);
    // number of ground truth bboxes of the three streams are: 4, 3, 6
    // respectively.
    bbox_gt_shape[0] = 4 + 3 + 6;
    bbox_gt_shape[1] = 5;
    bbox_gt_ = new Blob<Dtype>(bbox_gt_shape);
    filler.Fill(bbox_gt_);
    int batch_ind_array[] = {0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 2, 2};
    Dtype *bbox_gt_data = bbox_gt_->mutable_cpu_data();
    for (int i = 0; i < bbox_gt_->num(); ++i) {
      bbox_gt_data[0] = batch_ind_array[i];
      bbox_gt_data += bbox_gt_->offset(1);
    }

    vector<int> obj_score_shape(3);
    obj_score_shape[0] = 5;
    obj_score_shape[1] = 3;
    obj_score_shape[2] = 2;
    confidence_ = new Blob<Dtype>(obj_score_shape);
    // confidence_(0, :, :) =
    // 0.8147    0.9134
    // 0.9058    0.6324
    // 0.1270    0.0975
    // confidence_(1, :, :) =
    // 0.2785    0.9649
    // 0.5469    0.1576
    // 0.9575    0.9706
    // confidence_(2, :, :) =
    // 0.9572    0.1419
    // 0.4854    0.4218
    // 0.8003    0.9157
    // confidence_(3, :, :) =
    // 0.7922    0.0357
    // 0.9595    0.8491
    // 0.6557    0.9340
    // confidence_(4, :, :) =
    // 0.6787    0.3922
    // 0.7577    0.6555
    // 0.7431    0.1712
    Dtype obj_score[] = {0.8147, 0.9134, 0.9058, 0.6324, 0.1270, 0.0975,
                         0.2785, 0.9649, 0.5469, 0.1576, 0.9575, 0.9706,
                         0.9572, 0.1419, 0.4854, 0.4218, 0.8003, 0.9157,
                         0.7922, 0.0357, 0.9595, 0.8491, 0.6557, 0.9340,
                         0.6787, 0.3922, 0.7577, 0.6555, 0.7431, 0.1712};
    Dtype *obj_score_data = confidence_->mutable_cpu_data();
    for (int i = 0; i < confidence_->count(); ++i) {
      *obj_score_data++ = obj_score[i];
    }

    vector<int> labels_shape(1, 4 + 3 + 6);
    labels_ = new Blob<Dtype>(labels_shape);
    // labels(:) =
    // 14, 0, 5, 0, 2, 17, 14, 6, 19, 0, 9, 8, 9
    int labels_array[] = {14, 3, 5, 2, 2, 17, 14, 6, 19, 4, 9, 8, 9};
    Dtype *labels_data = labels_->mutable_cpu_data();
    for (int i = 0; i < labels_->count(); ++i) {
      *labels_data++ = labels_array[i];
    }

    vector<int> cls_score_shape(3);
    cls_score_shape[0] = 5;
    cls_score_shape[1] = 3;
    cls_score_shape[2] = 21;
    cls_scores_ = new Blob<Dtype>(cls_score_shape);
    filler.Fill(cls_scores_);

    bottom_vec_.push_back(bbox_pred_);
    bottom_vec_.push_back(bbox_gt_);
    bottom_vec_.push_back(labels_);
    bottom_vec_.push_back(confidence_);
    bottom_vec_.push_back(cls_scores_);
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
  vector<Blob<Dtype> *> bottom_vec_;
  vector<Blob<Dtype> *> top_vec_;
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

TYPED_TEST(HungarianLossLayerTest, TestPrepareForBBoxes) {
  typedef typename TypeParam::Dtype Dtype;

  LayerParameter layer_param;
  HungarianLossLayer<Dtype> layer(layer_param);
  layer.SetUp(this->bottom_vec_, this->top_vec_);
  layer.Forward(this->bottom_vec_, this->top_vec_);

  const vector<int> &num_gt_bboxes_vec = layer.num_gt_bboxes_vec();
  const vector<int> &start_idx_vec = layer.start_idx_vec();
  EXPECT_EQ(num_gt_bboxes_vec.size(), 3);
  EXPECT_EQ(num_gt_bboxes_vec[0], 4);
  EXPECT_EQ(num_gt_bboxes_vec[1], 3);
  EXPECT_EQ(num_gt_bboxes_vec[2], 6);
  EXPECT_EQ(start_idx_vec.size(), 3);
  EXPECT_EQ(start_idx_vec[0], 0);
  EXPECT_EQ(start_idx_vec[1], 4);
  EXPECT_EQ(start_idx_vec[2], 7);

  const vector<int> num_pred_bboxes = layer.num_pred_bboxes();
  EXPECT_EQ(num_pred_bboxes.size(), 3);
  EXPECT_EQ(num_pred_bboxes[0], 5);
  EXPECT_EQ(num_pred_bboxes[1], 4);
  EXPECT_EQ(num_pred_bboxes[2], 6);
  const vector<int> num_pred_bboxes_truth = layer.num_pred_bboxes_truth();
  EXPECT_EQ(num_pred_bboxes_truth.size(), 3);
  EXPECT_EQ(num_pred_bboxes_truth[0], 5);
  EXPECT_EQ(num_pred_bboxes_truth[1], 4);
  EXPECT_EQ(num_pred_bboxes_truth[2], 5);
}

TYPED_TEST(HungarianLossLayerTest, TestPrepareForConf) {
  typedef typename TypeParam::Dtype Dtype;

  LayerParameter layer_param;
  HungarianLossLayer<Dtype> layer(layer_param);
  layer.SetUp(this->bottom_vec_, this->top_vec_);
  layer.Forward(this->bottom_vec_, this->top_vec_);

  const vector<vector<Dtype> > &min_objectness = layer.min_objectness();
  EXPECT_EQ(min_objectness.size(), 3);
  Dtype min_obj_array[][5] = {0.5246, 0.5246, 0.3068, 0.3068, 0.3068,
                              0.4321, 0.4039, 0.4039, 0.4039, 0.4039,
                              0.4926, 0.4926, 0.4926, 0.4926, 0.3608};
  for (int i = 0; i < min_objectness.size(); ++i) {
    if (i == 1) {
      EXPECT_EQ(min_objectness[i].size(), 4);
    }
    else {
      EXPECT_EQ(min_objectness[i].size(), 5);
    }
    //std::cout << "i = " << i << std::endl;
    for (int j = 0; j < min_objectness[i].size(); ++j) {
      EXPECT_NEAR(min_obj_array[i][j], min_objectness[i][j], 1e-4);
      //std::cout << "\ti, j = " << i << ", " << j <<std::endl;
    }
  }

  const vector<vector<int> > &min_idx = layer.min_idx();
  EXPECT_EQ(min_idx.size(), 3);
  int min_idx_array[][5] = {0, 0, 2, 2, 2,
                            0, 1, 1, 1, 1,
                            0, 0, 0, 0, 4};
  for (int i = 0; i < min_idx.size(); ++i) {
    if (i == 1) {
      EXPECT_EQ(min_idx[i].size(), 4);
    }
    else {
      EXPECT_EQ(min_idx[i].size(), 5);
    }
    for (int j = 0; j < min_idx[i].size(); ++j) {
      EXPECT_EQ(min_idx_array[i][j], min_idx[i][j]);
    }
  }
}

TYPED_TEST(HungarianLossLayerTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;

  LayerParameter layer_param;
  layer_param.add_loss_weight(1);
  layer_param.mutable_hungarian_loss_param()->set_cls_weight(1);
  layer_param.mutable_hungarian_loss_param()->set_obj_weight(1);
  HungarianLossLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-2, 1701);
  //checker.CheckGradientExhaustive(&layer, this->bottom_vec_, this->top_vec_,
  //                                 0);
  //checker.CheckGradientExhaustive(&layer, this->bottom_vec_, this->top_vec_,
  //                                 3);
  checker.CheckGradientExhaustive(&layer, this->bottom_vec_, this->top_vec_, 4);
}

}  // namespace caffe