#ifndef CAFFE_HUNGARIAN_LOSS_LAYER_HPP_
#define CAFFE_HUNGARIAN_LOSS_LAYER_HPP_

#include "caffe/loss_layers.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

template <typename Dtype>
class HungarianLossLayer : public LossLayer<Dtype> {
 public:
  explicit HungarianLossLayer(const LayerParameter &param) :
      LossLayer<Dtype>(param) { }
  virtual void LayerSetUp(const vector<Blob<Dtype>*> &bottom,
                          const vector<Blob<Dtype>*> &top);
  virtual void Reshape(const vector<Blob<Dtype>*> &bottom,
                       const vector<Blob<Dtype>*> &top);

  virtual inline const char *type() const { return "HungarianLoss";}
  virtual inline int ExactNumBottomBlobs() const { return 5; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*> &bottom,
                           const vector<Blob<Dtype>*> &top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*> &top,
                            const vector<bool> &propagate_down,
                            const vector<Blob<Dtype>*> &bottom);
  // const vector<int> &assignment() const { return assignment_; }

  // loss weight for objectness confidence.
  float obj_weight_;
  // loss weight for classification score.
  float cls_weight_;
  // assignment_[i] is the index of the assigned ground truth bbox of the
  // ith predicted bbox.
  vector<int> assignment_;
  // min_objectness[i] is the smallest among s0, s1, ..., si,
  // where si is the objectness confidence of the ith predicted 
  // bbox, and min_idx_[i] is the index of the smallest sj(0<= j <= i).
  vector<Dtype> min_objectness_;
  vector<int> min_idx_;
  // internel softmax layer for objectness confidence to normalize the
  // objectness confidence.
  shared_ptr<Layer<Dtype> > softmax_obj_layer_;
  vector<Blob<Dtype>*> softmax_obj_bottom_vec_;
  vector<Blob<Dtype>*> softmax_obj_top_vec_;
  Blob<Dtype> confidence_;
  // internel softmax layer for classification scores to normalize the
  // classification scores.
  shared_ptr<Layer<Dtype> > softmax_cls_layer_;
  vector<Blob<Dtype>*> softmax_cls_bottom_vec_;
  vector<Blob<Dtype>*> softmax_cls_top_vec_;
  Blob<Dtype> prob_;

  int num_pred_bboxes_;
  int num_pred_bboxes_truth_;
};

}  // namespace caffe

#endif  // CAFFE_HUNGARIAN_LOSS_LAYER_HPP_