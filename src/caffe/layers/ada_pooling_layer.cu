#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/ada_pooling_layer.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"

using std::floor;
using std::ceil;
using std::min;
using std::max;

namespace caffe {

template <typename Dtype>
__global__ void MaxPoolForward(const int nthreads, const Dtype *bottom_data,
    const int num, const int channels, const int height,
    const int width, const int pooled_height, const int pooled_width,
    Dtype *top_data, int *argmax_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int c = (index / pooled_width / pooled_height) % channels;
    int n = index / pooled_width / pooled_height / channels;

    Dtype bin_size_h = static_cast<Dtype>(height) / pooled_height;
    Dtype bin_size_w = static_cast<Dtype>(width) / pooled_width;
    int hstart = static_cast<int>(floor(ph * bin_size_h));
    int wstart = static_cast<int>(floor(pw * bin_size_w));
    int hend = static_cast<int>(ceil((ph + 1) * bin_size_h));
    int wend = static_cast<int>(ceil((pw + 1) * bin_size_w));

    hstart = min(max(0, hstart), height);
    hend = min(max(0, hend), height);
    wstart = min(max(0, wstart), width);
    wend = min(max(0, wend), width);
    bool is_empty = (hend <= hstart) || (wstart) <= wend;

    // Define an empty pooling region to be zero.
    Dtype maxval = is_empty ? 0 : -FLT_MAX;
    // If nothing is pooled, argmax = -1 causes nothing to be backpropagated.
    int maxidx = -1;
    bottom_data += (n * channels + c) * height * width;
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        int bottom_index = h * width + w;
        if (bottom_data[bottom_index] > maxval) {
          maxval = bottom_data[bottom_index];
          maxidx = bottom_index;
        }
      }
    }
    top_data[index] = maxval;
    argmax_data[index] = maxidx;
  }
}

template <typename Dtype>
void AdaPoolingLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*> &bottom,
                                         const vector<Blob<Dtype>*> &top) {
  const Dtype *bottom_data = bottom[0]->gpu_data();
  Dtype *top_data = top[0]->mutable_gpu_data();
  int *argmax_data = max_idx_.mutable_gpu_data();
  int count = top[0]->count();
  MaxPoolForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, bottom_data, bottom[0]->num(), channels_, height_, width_,
      pooled_height_, pooled_width_, top_data, argmax_data);
  CUDA_POST_KERNEL_CHECK;
}

template <typename Dtype>
__global__ void MaxPoolBackward(const int nthreads, const Dtype *top_diff,
    const int *argmax_data, const int num, const int channels,
    const int height, const int width, const int pooled_height,
    const int pooled_width, Dtype *bottom_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    // (n, c, h, w) coords in bottom data
    int w = index % width;
    int h = (index / width) % height;
    int c = (index / width / height) % channels;
    int n = index / width / height / channels;

    Dtype bin_size_h = static_cast<Dtype>(height) / pooled_height;
    Dtype bin_size_w = static_cast<Dtype>(width) / pooled_width;

    int phstart = floor(h / bin_size_h);
    int phend = ceil((h + 1) / bin_size_h);
    int pwstart = floor(w / bin_size_w);
    int pwend = ceil((w + 1) / bin_size_w);

    phstart = min(max(phstart, 0), pooled_height);
    phend = min(max(phend, 0), pooled_height);
    pwstart = min(max(pwstart, 0), pooled_width);
    pwend = min(max(pwend, 0), pooled_width);

    Dtype gradient = 0;
    int offset = (n * channels + c) * pooled_height * pooled_width;
    top_diff += offset;
    argmax_data += offset;
    for (int ph = phstart; ph < phend; ++ph) {
      for (int pw = pwstart; pw < pwend; ++pw) {
        if (argmax_data[ph * pooled_width + pw] == h*width + w) {
          gradient += top_diff[ph * pooled_width + pw];
        }
      }
    }
    bottom_diff[index] = gradient;
  }
}

template <typename Dtype>
void AdaPoolingLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*> &top,
                                          const vector<bool> &propagate_down,
                                          const vector<Blob<Dtype>*> &bottom) {
  if (!propagate_down[0]) {
    return;
  }
  const Dtype *top_diff = top[0]->gpu_diff();
  Dtype *bottom_diff = bottom[0]->mutable_gpu_diff();
  const int count = bottom[0]->count();
  caffe_gpu_set(count, Dtype(0), bottom_diff);
  const int *argmax_data = max_idx_.gpu_data();
  MaxPoolBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, top_diff, argmax_data, top[0]->num(), channels_, height_, width_,
      pooled_height_, pooled_width_, bottom_diff);
  CUDA_POST_KERNEL_CHECK;
}

INSTANTIATE_LAYER_GPU_FUNCS(AdaPoolingLayer);

}  // namespace caffe