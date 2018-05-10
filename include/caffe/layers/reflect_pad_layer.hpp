#ifndef CAFFE_REFLECT_PAD_LAYER_HPP_
#define CAFFE_REFLECT_PAD_LAYER_HPP_

#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

	/**
	* @brief Takes a Blob and crop it, to the shape specified by the second input
	*  Blob, across all dimensions after the specified axis.
	*
	* TODO(dox): thorough documentation for Forward, Backward, and proto params.
	*/

	template <typename Dtype>
	class ReflectPaddingLayer : public Layer<Dtype> {
	public:
		explicit ReflectPaddingLayer(const LayerParameter& param)
			: Layer<Dtype>(param) {
				if (param.has_reflect_pad_param())
				{
					pad_h = param.reflect_pad_param().pad_h();
					pad_w = param.reflect_pad_param().pad_w();
				}
				else
				{
					pad_h = pad_w = 0;
				}
			}
		virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);

		virtual inline const char* type() const { return "ReflectPadding"; }

	protected:
		virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
		virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

		unsigned pad_h, pad_w;

	};
}  // namespace caffe

#endif
