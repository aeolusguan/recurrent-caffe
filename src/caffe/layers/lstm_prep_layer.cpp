//
// Created by aeolus on 16-3-23.
//

#include "caffe/lstm_prep_layer.hpp"
#include "caffe/layer_factory.hpp"

namespace caffe {

template <typename Dtype>
string LSTMPrepLayer<Dtype>::int_to_str(const int t) const {
  ostringstream num;
  num << t;
  return num.str();
}

template <typename Dtype>
void LSTMPrepLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*> &bottom,
                                      const vector<Blob<Dtype>*> &top) {
  NetParameter net_param;
  net_param.set_force_backward(true);

  net_param.add_input("single_x");
  BlobShape input_shape;
  for (int i = 0; i < bottom[0]->num_axes(); ++i) {
    input_shape.add_dim(bottom[0]->shape(i));
  }
  net_param.add_input_shape()->CopyFrom(input_shape);
  T_ = 10;

  // SplitLayer
  LayerParameter *split_param = net_param.add_layer();
  split_param->set_type("Split");
  split_param->set_name("single_x_split");
  split_param->add_bottom("single_x");

  // ConcatLayer
  LayerParameter *concat_param = net_param.add_layer();
  concat_param->set_type("Concat");
  concat_param->set_name("single_x_concat");
  concat_param->mutable_concat_param()->set_axis(0);
  concat_param->add_top("stacked_x");

  for (int i = 1; i <= T_; ++i) {
    string ts = int_to_str(i);
    split_param->add_top("single_x_" + ts);
    concat_param->add_bottom("single_x_" + ts);
  }

  // Create the net.
  net_.reset(new Net<Dtype>(net_param));

  single_x_input_ = net_->blob_by_name("single_x");
  stacked_x_output_ = net_->blob_by_name("stacked_x");
}

template <typename Dtype>
void LSTMPrepLayer<Dtype>::Reshape(const vector<Blob<Dtype>*> &bottom,
                                   const vector<Blob<Dtype>*> &top) {
  CHECK_EQ(bottom[0]->num(), 1) << "bottom[1] should be a single timestep";

  BlobShape output_shape;
  output_shape.add_dim(T_);
  for (int i = 1; i < bottom[0]->num_axes(); ++i) {
    output_shape.add_dim(bottom[0]->shape(i));
  }
  top[0]->Reshape(output_shape);

  output_shape.Clear();
  output_shape.add_dim(T_);
  output_shape.add_dim(1);
  top[1]->Reshape(output_shape);

  single_x_input_->ShareData(*bottom[0]);
  single_x_input_->ShareDiff(*bottom[0]);
  stacked_x_output_->ShareData(*top[0]);
  stacked_x_output_->ShareDiff(*top[0]);
}

template <typename Dtype>
void LSTMPrepLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*> &bottom,
                                       const vector<Blob<Dtype>*> &top) {
  net_->ForwardPrefilled();
  Dtype *cont_data = top[1]->mutable_cpu_data();
  for (int i = 1; i < T_; ++i) {
    cont_data[i] = Dtype(1);
  }
  cont_data[0] = Dtype(0);
}

template <typename Dtype>
void LSTMPrepLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*> &top,
                                        const vector<bool> &propagate_down,
                                        const vector<Blob<Dtype>*> &bottom) {
  net_->Backward(); 
}

#ifdef CPU_ONLY
STUB_GPU(LSTMPrepLayer);
#endif

INSTANTIATE_CLASS(LSTMPrepLayer);
REGISTER_LAYER_CLASS(LSTMPrep);

}  // namespace caffe