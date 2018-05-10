#ifndef CAFFE_NEAREST_SAMPLE_LAYER_HPP_
#define CAFFE_NEAREST_SAMPLE_LAYER_HPP_

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
	class NearestSampleLayer : public Layer<Dtype> {
	public:
		explicit NearestSampleLayer(const LayerParameter& param)
			: Layer<Dtype>(param) {
				if (param.has_nearest_neighbor_param())
				{
					scale = param.nearest_neighbor_param().scale();
				}
				else
				{
					scale = 2;
				}
			}
		virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);

		virtual inline const char* type() const { return "NearestSample"; }

	protected:
		virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
		virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

		float scale;

	};
}  // namespace caffe

#endif
