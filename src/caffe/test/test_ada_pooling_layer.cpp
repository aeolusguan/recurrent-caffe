#include <cstring>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/ada_pooling_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

template <typename TypeParam>
class AdaPoolingLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  AdaPoolingLayerTest() :
      blob_bottom_(new Blob<Dtype>(2, 2, 14, 14)),
      blob_top_(new Blob<Dtype>()),
      blob_argmax_(new Blob<Dtype>()) { 
    Caffe::set_random_seed(1701);
    // Fill the values.
    FillerParameter filler_param;
    filler_param.set_std(10);
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
  }
  virtual ~AdaPoolingLayerTest() {
    delete blob_bottom_;
    delete blob_top_;
    delete blob_argmax_;
  }
  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_;
  Blob<Dtype>* const blob_argmax_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(AdaPoolingLayerTest, TestDtypesAndDevices);

TYPED_TEST(AdaPoolingLayerTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  AdaPoolingParameter *ada_pool_param = 
      layer_param.mutable_ada_pooling_param();
  ada_pool_param->set_pooled_h(7);
  ada_pool_param->set_pooled_w(7);
  AdaPoolingLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-4, 1e-2);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_, 0);
}

}  // namespace caffe