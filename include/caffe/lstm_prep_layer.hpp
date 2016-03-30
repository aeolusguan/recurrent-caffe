//
// Created by aeolus on 16-3-23.
//

#ifndef CAFFE_LSTM_PREP_HPP
#define CAFFE_LSTM_PREP_HPP

#include "caffe/layer.hpp"
#include "caffe/net.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

// Layer to prepare input blobs for LSTM layer.
template<typename Dtype>
class LSTMPrepLayer: public Layer<Dtype> {
 public:
  LSTMPrepLayer(const LayerParameter &param) :
      Layer<Dtype>(param) { }
  virtual void LayerSetUp(const vector<Blob<Dtype>*> &bottom,
                          const vector<Blob<Dtype>*> &top);
  virtual void Reshape(const vector<Blob<Dtype>*> &bottom,
                       const vector<Blob<Dtype>*> &top);

  virtual inline const char *type() const { return "LSTMPrep"; }
  virtual inline int ExactNumBottomBlobs() const { return 1; }
  virtual inline int ExactNumTopBlobs() const { return 2; }
  virtual inline bool AllowForceBackward(const int bottom_index) const {
    // Can't propagate to ground truth label.
    return bottom_index != 0;
  }
 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*> &bottom,
                           const vector<Blob<Dtype>*> &top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*> &top,
                            const vector<bool> &propagate_down,
                            const vector<Blob<Dtype>*> &bottom);

  // A helper function, useful for stringifying timestep indices.
  string int_to_str(const int t) const;

  // A Net to implement the tile functionality.
  shared_ptr<Net<Dtype> > net_;

  // The number of timesteps in the layer's output.
  int T_;
  shared_ptr<Blob<Dtype> > single_x_input_;
  shared_ptr<Blob<Dtype> > stacked_x_output_;
};

}  // namespace caffe

#endif //CAFFE_LSTM_PREP_HPP
