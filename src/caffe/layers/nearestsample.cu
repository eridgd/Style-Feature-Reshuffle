#include <vector>

#include "caffe/layers/nearestsample.hpp"

namespace caffe {

	template <typename Dtype>
	__global__ void UpSample(Dtype* input, Dtype*output, float scale,dim3 inputdim,dim3 outputdim)
	{
		int outputPointId = threadIdx.x + blockIdx.x * blockDim.x;
		if (outputPointId > outputdim.z*outputdim.y*outputdim.x) return;
		int outputPointX = outputPointId % outputdim.z;
		int outputPointY = (outputPointId / outputdim.z) % outputdim.y;
		int channel = (outputPointId / outputdim.z) / outputdim.y;
		int inputPointX = outputPointX / scale;

		int inputPointY = outputPointY / scale;

		size_t offset_i = (channel*inputdim.y + inputPointY)*inputdim.z + inputPointX;
		size_t offset_o = (channel*outputdim.y + outputPointY)*outputdim.z + outputPointX;
		output[offset_o] = input[offset_i];
	}

	template <typename Dtype>
	void NearestSampleLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype> * >& bottom,
		const vector<Blob<Dtype> * >& top)
	{
		const int count = top[0]->channels()*top[0]->height()*top[0]->width();
		UpSample<Dtype> << <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS >> >(bottom[0]->mutable_gpu_data(), top[0]->mutable_gpu_data(), scale, dim3(bottom[0]->channels(), bottom[0]->height(), bottom[0]->width()), dim3(top[0]->channels(), top[0]->height(), top[0]->width()));
		CUDA_POST_KERNEL_CHECK;
	}

	template <typename Dtype>
	void NearestSampleLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom)
	{
		NOT_IMPLEMENTED;
	}

	INSTANTIATE_LAYER_GPU_FUNCS(NearestSampleLayer);
}  // namespace caffe
