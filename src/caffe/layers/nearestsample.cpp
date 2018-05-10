#include <algorithm>
#include <functional>
#include <map>
#include <set>
#include <vector>


#include "caffe/layer.hpp"
#include "caffe/layers/nearestsample.hpp"
#include "caffe/net.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

	template <typename Dtype>
	void NearestSampleLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		top[0]->Reshape(bottom[0]->num(), bottom[0]->channels(), bottom[0]->height() * scale, bottom[0]->width() * scale);
	}
	template <typename Dtype>
	void NearestSampleLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top)
	{
		NOT_IMPLEMENTED;
	}
	template <typename Dtype>
	void NearestSampleLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom)
	{
		NOT_IMPLEMENTED;
	}
	INSTANTIATE_CLASS(NearestSampleLayer);
	REGISTER_LAYER_CLASS(NearestSample);
}