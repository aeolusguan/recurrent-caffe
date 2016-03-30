#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/filler.hpp"
#include "caffe/lstm_prep_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

template <typename TypeParam>
class LSTMPrepLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;
 protected:
  LSTMPrepLayerTest() :
      stacked_x_blob_(new Blob<Dtype>()),
      cont_blob_(new Blob<Dtype>()) {
    std::vector<int> x_input_shape(3);
    x_input_shape[0] = 1;
    x_input_shape[1] = 1;
    x_input_shape[2] = 100;
    single_x_blob_ = new Blob<Dtype>(x_input_shape);
    FillerParameter filler_param;
    filler_param.set_std(10);
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(single_x_blob_);

    blob_bottom_vec_.push_back(single_x_blob_);
    blob_top_vec_.push_back(stacked_x_blob_);
    blob_top_vec_.push_back(cont_blob_);
  }
  virtual ~LSTMPrepLayerTest() {
    delete single_x_blob_;
    delete stacked_x_blob_;
    delete cont_blob_;
  }
  Blob<Dtype> *single_x_blob_;
  Blob<Dtype> *stacked_x_blob_;
  Blob<Dtype> *cont_blob_;
  std::vector<Blob<Dtype>*> blob_bottom_vec_;
  std::vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(LSTMPrepLayerTest, TestDtypesAndDevices);

TYPED_TEST(LSTMPrepLayerTest, TestSetUp) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  LSTMPrepLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  vector<int> expected_top_shape = this->single_x_blob_->shape();
  expected_top_shape[0] = 10;
  EXPECT_TRUE(this->stacked_x_blob_->shape() == expected_top_shape);
}

TYPED_TEST(LSTMPrepLayerTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  LSTMPrepLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-2, 1701);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_, 0);
}

}  // namespace caffe