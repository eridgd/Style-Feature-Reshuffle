#include <vector>
#include "caffe/layers/reflect_pad_layer.hpp"

namespace caffe {

	template <typename Dtype>
	__global__ void Reflect_Forward(Dtype* input, Dtype* output, int pad_h, int pad_w, dim3 inputdim, dim3 outputdim)
	{
		int outputPointId = threadIdx.x + blockIdx.x * blockDim.x;
		if (outputPointId >= outputdim.z*outputdim.y*outputdim.x) return;
		int outputPointX = outputPointId % outputdim.z;
		int outputPointY = (outputPointId / outputdim.z) % outputdim.y;
		int channel = (outputPointId / outputdim.z) / outputdim.y;
		
		int iStartX = max(0, -pad_w);
		int iStartY = max(0, -pad_h);
		int oStartX = max(0, pad_w);
		int oStartY = max(0, pad_h);

		int inputPointX = abs(outputPointX - pad_w)
			- abs(outputPointX - (int)(inputdim.z + pad_w - 1))
			- outputPointX
			+ 2 * pad_w + inputdim.z - 1
			- oStartX + iStartX;

		int inputPointY = abs(outputPointY - pad_h)
			- abs(outputPointY - (int)(inputdim.y + pad_h - 1))
			- outputPointY
			+ 2 * pad_h + inputdim.y - 1
			- oStartY + iStartY;

		size_t offset_i = (channel*inputdim.y + inputPointY)*inputdim.z + inputPointX;
		size_t offset_o = (channel*outputdim.y + outputPointY)*outputdim.z + outputPointX;
		output[offset_o] = input[offset_i];
	}
	template <typename Dtype>
	void ReflectPaddingLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
	{
		const int count = top[0]->channels()*top[0]->height()*top[0]->width();
		Reflect_Forward<Dtype> << <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS >> >(bottom[0]->mutable_gpu_data(), top[0]->mutable_gpu_data(), pad_h, pad_w, dim3(bottom[0]->channels(), bottom[0]->height(), bottom[0]->width()), dim3(top[0]->channels(), top[0]->height(), top[0]->width()));
		CUDA_POST_KERNEL_CHECK;
	}

	template <typename Dtype>
	void ReflectPaddingLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom)
	{
		NOT_IMPLEMENTED;
	}
	INSTANTIATE_LAYER_GPU_FUNCS(ReflectPaddingLayer);
}  // namespace caffe
