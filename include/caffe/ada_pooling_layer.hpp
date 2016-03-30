#ifndef CAFFE_ADA_POOLING_LAYER_HPP_
#define CAFFE_ADA_POOLING_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

// AdaPoolingLayer - Adaptive Pooling Layer
template <typename Dtype>
class AdaPoolingLayer : public Layer<Dtype> {
 public:
  explicit AdaPoolingLayer(const LayerParameter &param) :
      Layer<Dtype>(param) { }
  virtual void LayerSetUp(const vector<Blob<Dtype>*> &bottom,
                          const vector<Blob<Dtype>*> &top);
  virtual void Reshape(const vector<Blob<Dtype>*> &bottom,
                       const vector<Blob<Dtype>*> &top);

  virtual inline const char *type() const { return "AdaPooling"; }
  virtual inline int ExactBottomBlobs() const { return 1; }
  virtual inline int ExactTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*> &bottom,
                           const vector<Blob<Dtype>*> &top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*> &bottom,
                           const vector<Blob<Dtype>*> &top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*> &top,
                            const vector<bool> &propagate_down,
                            const vector<Blob<Dtype>*> &bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*> &top,
                            const vector<bool> &propagate_down,
                            const vector<Blob<Dtype>*> &bottom);

  int channels_;
  int height_;
  int width_;
  int pooled_height_;
  int pooled_width_;
  Blob<int> max_idx_;
};

}  // namespace caffe

#endif  // CAFFE_ADA_POOLING_LAYER_HPP_