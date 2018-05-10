#include <algorithm>
#include <functional>
#include <map>
#include <set>
#include <vector>


#include "caffe/layer.hpp"
#include "caffe/layers/reflect_pad_layer.hpp"
#include "caffe/net.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

	template <typename Dtype>
	void ReflectPaddingLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		top[0]->Reshape(bottom[0]->num(), bottom[0]->channels(), bottom[0]->height() + 2 * pad_h, bottom[0]->width() + 2 * pad_w);
	}
	template <typename Dtype>
	void ReflectPaddingLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top)
	{
		NOT_IMPLEMENTED;
	}
	template <typename Dtype>
	void ReflectPaddingLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom)
	{
		NOT_IMPLEMENTED;
	}

	INSTANTIATE_CLASS(ReflectPaddingLayer);
	REGISTER_LAYER_CLASS(ReflectPadding);

}