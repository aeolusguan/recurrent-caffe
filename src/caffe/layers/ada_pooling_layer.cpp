#include <cfloat>

#include "caffe/ada_pooling_layer.hpp"
#include "caffe/util/math_functions.hpp"

using std::floor;
using std::ceil;
using std::min;
using std::max;

namespace caffe {

template <typename Dtype>
void AdaPoolingLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*> &bottom,
                                        const vector<Blob<Dtype>*> &top) {
  AdaPoolingParameter ada_pool_param = this->layer_param_.ada_pooling_param();
  CHECK_GT(ada_pool_param.pooled_h(), 0)
      << "pooled_h must be > 0";
  CHECK_GT(ada_pool_param.pooled_w(), 0)
      << "pooled_w must be > 0";
  pooled_height_ = ada_pool_param.pooled_h();
  pooled_width_ = ada_pool_param.pooled_w();
}      

template <typename Dtype>
void AdaPoolingLayer<Dtype>::Reshape(const vector<Blob<Dtype>*> &bottom,
                                     const vector<Blob<Dtype>*> &top) {
  channels_ = bottom[0]->channels();
  height_ = bottom[0]->height();
  width_ = bottom[0]->width();
  top[0]->Reshape(bottom[0]->num(), channels_, pooled_height_, pooled_width_);
  max_idx_.Reshape(bottom[0]->num(), channels_, pooled_height_, pooled_width_);
}

template <typename Dtype>
void AdaPoolingLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*> &bottom,
                                         const vector<Blob<Dtype>*> &top) {
  const Dtype *bottom_data = bottom[0]->cpu_data();
  Dtype *top_data = top[0]->mutable_cpu_data();
  const int top_count = top[0]->count();
  caffe_set(top_count, Dtype(-FLT_MAX), top_data);
  int *argmax_data = max_idx_.mutable_cpu_data();
  caffe_set(top_count, -1, argmax_data);
  // The main loop
  const Dtype bin_size_h = static_cast<Dtype>(height_) / pooled_height_;
  const Dtype bin_size_w = static_cast<Dtype>(width_) / pooled_width_;
  for (int n = 0; n < bottom[0]->num(); ++n) {
    for (int c = 0; c < channels_; ++c) {
      for (int ph = 0; ph < pooled_height_; ++ph) {
        for (int pw = 0; pw < pooled_width_; ++pw) {
          // Compute pooling region for this output unit:
          //  start (included) = floor(ph * height_ / pooled_height_)
          //  end (excluded) = ceil((ph + 1) * height / pooled_height_)
          int hstart = static_cast<int>(floor(ph * bin_size_h));
          int wstart = static_cast<int>(floor(pw * bin_size_w));
          int hend = static_cast<int>(ceil((ph + 1) * bin_size_h));
          int wend = static_cast<int>(ceil((pw + 1) * bin_size_w));

          hstart = min(max(0, hstart), height_);
          hend = min(max(0, hend), height_);
          wstart = min(max(0, wstart), width_);
          wend = min(max(0, wend), width_);

          bool is_empty = (hend <= hstart) || (wend <= wstart);

          const int pool_index = ph * pooled_width_ + pw;
          if (is_empty) {
            top_data[pool_index] = 0;
            argmax_data[pool_index] = -1;
          }

          for (int h = hstart; h < hend; ++h) {
            for (int w = wstart; w < wend; ++w) {
              const int index = h * width_ + w;
              if (bottom_data[index] > top_data[pool_index]) {
                top_data[pool_index] = bottom_data[index];
                argmax_data[pool_index] = index;
              }
            }
          }
        }
      }
      // Increment all data pointers by one channel.
      bottom_data += bottom[0]->offset(0, 1);
      top_data += top[0]->offset(0, 1);
      argmax_data += top[0]->offset(0, 1);
    }
  }
}

template <typename Dtype>
void AdaPoolingLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*> &top,
                                          const vector<bool> &propagate_down,
                                          const vector<Blob<Dtype>*> &bottom) {
  if (!propagate_down[0]) {
    return;
  }
  const Dtype *top_diff = top[0]->cpu_diff();
  Dtype *bottom_diff = bottom[0]->mutable_cpu_diff();
  caffe_set(bottom[0]->count(), Dtype(0), bottom_diff);
  const int *argmax_data = max_idx_.cpu_data();
  // The main loop
  for (int n = 0; n < top[0]->num(); ++n) {
    for (int c = 0; c < channels_; ++c) {
      for (int ph = 0; ph < pooled_height_; ++ph) {
        for (int pw = 0; pw < pooled_width_; ++pw) {
          const int index = ph * pooled_width_ + pw;
          const int bottom_index = argmax_data[index];
          // TODO: justify whether bottom_index == -1
          if (bottom_index == -1) continue;
          bottom_diff[bottom_index] += top_diff[index];
        }
      }
      bottom_diff += bottom[0]->offset(0, 1);
      top_diff += top[0]->offset(0, 1);
      argmax_data += top[0]->offset(0, 1);
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(AdaPoolingLayer)
#endif

INSTANTIATE_CLASS(AdaPoolingLayer);
REGISTER_LAYER_CLASS(AdaPooling);

}  // namespace caffe