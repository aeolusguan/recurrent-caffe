#ifndef CAFFE_HUNGARIAN_LOSS_LAYER_HPP_
#define CAFFE_HUNGARIAN_LOSS_LAYER_HPP_

#include "caffe/loss_layers.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

template<typename Dtype>
class HungarianLossLayer: public LossLayer<Dtype> {
 public:
  explicit HungarianLossLayer(const LayerParameter &param) :
      LossLayer<Dtype>(param) { }
  virtual void LayerSetUp(const vector<Blob<Dtype> *> &bottom,
                          const vector<Blob<Dtype> *> &top);
  virtual void Reshape(const vector<Blob<Dtype> *> &bottom,
                       const vector<Blob<Dtype> *> &top);

  virtual inline const char *type() const { return "HungarianLoss"; }
  virtual inline int ExactNumBottomBlobs() const { return 5; }

  // Interface for test.
  const vector<int> &num_gt_bboxes_vec() const { return num_gt_bboxes_vec_; }
  const vector<int> &start_idx_vec() const { return start_idx_vec_; }
  const vector<int> &num_pred_bboxes() const { return num_pred_bboxes_; }
  const vector<int> &num_pred_bboxes_truth() const {
    return num_pred_bboxes_truth_;
  }
  const vector<vector<Dtype> > &min_objectness() const {
    return min_objectness_;
  }
  const vector<vector<int> > &min_idx() const { return min_idx_; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype> *> &bottom,
                           const vector<Blob<Dtype> *> &top);
  virtual void Backward_cpu(const vector<Blob<Dtype> *> &top,
                            const vector<bool> &propagate_down,
                            const vector<Blob<Dtype> *> &bottom);

  void PrepareForBBoxes(const Blob<Dtype> *gt_bbox_blob, int num_pred_bboxes);
  void PrepareForConf();
  Dtype ForwardBatch(const vector<Blob<Dtype> *> &bottom, int batch_ind);
  void BackwardBatch(const vector<Blob<Dtype> *> &bottom, int batch_ind);

  template <typename T>
  void ClearVec(vector<vector<T> > &vec) {
    for (int i = 0; i < vec.size(); ++i) {
      vec[i].clear();
    }
    vec.clear();
  }

  // const vector<int> &assignment() const { return assignment_; }

  // loss weight for objectness confidence.
  float obj_weight_;
  // loss weight for classification score.
  float cls_weight_;
  // assignment_[i][j] is the index of the assigned ground truth bbox of the
  // jth predicted bbox in the ith batch.
  vector<vector<int> > assignment_;
  // min_objectness[i][j] is the smallest among s0, s1, ..., sj,
  // where sj is the objectness confidence of the jth predicted 
  // bbox in the ith batch, and min_idx_[i][j] is the index of
  // the smallest in the ith batch.
  vector<vector<Dtype> > min_objectness_;
  vector<vector<int> > min_idx_;
  // internel softmax layer for objectness confidence to normalize the
  // objectness confidence.
  shared_ptr<Layer<Dtype> > softmax_obj_layer_;
  vector<Blob<Dtype> *> softmax_obj_bottom_vec_;
  vector<Blob<Dtype> *> softmax_obj_top_vec_;
  Blob<Dtype> confidence_;
  // internel softmax layer for classification scores to normalize the
  // classification scores.
  shared_ptr<Layer<Dtype> > softmax_cls_layer_;
  vector<Blob<Dtype> *> softmax_cls_bottom_vec_;
  vector<Blob<Dtype> *> softmax_cls_top_vec_;
  Blob<Dtype> prob_;

  // num_pred_bboxes_[i] is the height (and width) of the
  // hungarian cost matrix for the ith batch.
  vector<int> num_pred_bboxes_;
  // num_pred_bboxes_truth_[i] is the the number of the "real"
  // predicted bboxes to be assigned in the ith batch, whereas
  // "virtual" predicted bbox are padded.
  vector<int> num_pred_bboxes_truth_;
  // sum of all elements of num_pred_bboxes_truth_.
  int total_bboxes_;
  int batch_size_;
  // num_gt_bboxes_vec_[i]: number of ground truth bboxes of the ith batch.
  vector<int> num_gt_bboxes_vec_;
  // start_idx_vec_[i]: start index of the ith batch in ground truth bboxes.
  vector<int> start_idx_vec_;
};

}  // namespace caffe

#endif  // CAFFE_HUNGARIAN_LOSS_LAYER_HPP_