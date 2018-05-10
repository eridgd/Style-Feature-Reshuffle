

#include "GeneralizedPatchMatch.cuh"
#include "curand_kernel.h"

__host__ __device__ int clamp(int x, int x_max, int x_min) {//assume x_max >= x_min
	if (x > x_max)
	{
		return x_max;
	}
	else if (x < x_min)
	{
		return x_min;
	}
	else
	{
		return x;
	}
}

__host__ __device__ float clamp_f(float x, float x_max, float x_min) {//assume x_max >= x_min
	if (x > x_max)
	{
		return x_max;
	}
	else if (x < x_min)
	{
		return x_min;
	}
	else
	{
		return x;
	}
}

__host__ __device__ unsigned int XY_TO_INT(int x, int y) {//r represent the number of 10 degree, x,y - 11 bits, max = 2047, r - max = 36, 6 bits
	return (((y) << 11) | (x));
}
__host__ __device__ int INT_TO_X(unsigned int v) {
	return (v)&((1 << 11) - 1);
}
__host__ __device__ int INT_TO_Y(unsigned int v) {
	return (v >> 11)&((1 << 11) - 1);
}

__host__ __device__ int cuMax(int a, int b) {
	if (a > b) {
		return a;
	}
	else {
		return b;
	}
}
__host__ __device__ int cuMin(int a, int b) {
	if (a < b) {
		return a;
	}
	else {
		return b;
	}
}

__device__ float MycuRand(curandState &state) {//random number in cuda, between 0 and 1
	
	 return curand_uniform(&state);

}
__device__ void InitcuRand(curandState &state) {//random number in cuda, between 0 and 1
	
	int i = threadIdx.x + blockIdx.x * blockDim.x;

    curand_init(i, 0, 0, &state);

}



__host__ __device__ float dist_compute(float * a, float * b, int channels, int a_rows, int a_cols, int b_rows, int b_cols, int ax, int ay, int bx, int by, int patch_w) {//this is the average number of all matched pixel
	//suppose patch_w is an odd number
	double pixel_sum = 0, pixel_no = 0, pixel_dist = 0;//number of pixels realy counted
	double pixel_sum1 = 0;
	int a_slice = a_rows*a_cols, b_slice = b_rows*b_cols;
	int a_pitch = a_cols, b_pitch = b_cols;
	double dp_tmp;

	for (int dy = -patch_w / 2; dy <= patch_w / 2; dy++) {
		for (int dx = -patch_w / 2; dx <= patch_w / 2; dx++) {

			if (
				(ay + dy) < a_rows && (ay + dy) >= 0 && (ax + dx) < a_cols && (ax + dx) >= 0
				&&
				(by + dy) < b_rows && (by + dy) >= 0 && (bx + dx) < b_cols && (bx + dx) >= 0
				)//the pixel in a should exist and pixel in b should exist
			{
				if (channels == 3)
				{
					for (int dc = 0; dc < channels; dc++)
					{
						dp_tmp = a[dc * a_slice + (ay + dy) * a_pitch + (ax + dx)] - b[dc * b_slice + (by + dy) * b_pitch + (bx + dx)];
						pixel_sum += dp_tmp * dp_tmp;
					}
				}
				else
				{
					double dp_tmp = 0;
					for (int dc = 0; dc < channels; dc++)
					{
						dp_tmp += a[dc * a_slice + (ay + dy) * a_pitch + (ax + dx)] * b[dc * b_slice + (by + dy) * b_pitch + (bx + dx)];
					}

					pixel_sum -= dp_tmp;
				}


				pixel_no += 1;
			}
		}

	}


	if (pixel_no>0) pixel_dist = (pixel_sum + pixel_sum1) / pixel_no;
	else pixel_dist = 2.;
	//printf("dist:: ar:%d aw:%d br:%d bw:%d ax:%d ay:%d bx:%d by:%d dist:%.5lf\n", a_rows, a_cols, b_rows, b_cols, ax, ay, bx, by, pixel_dist);
	return pixel_dist;

}



__host__ __device__ float dist(float *a, float * b, int channels, int a_rows, int a_cols, int b_rows, int b_cols, int ax, int ay, int xp, int yp, int patch_w) {
	double d, x_diff, y_diff;
	d = dist_compute(a, b, channels, a_rows, a_cols, b_rows, b_cols, ax, ay, xp, yp, patch_w);
	return d;
}



__global__ void reInitialAnn_kernel(unsigned int * ann, int * params, float *local_cor_map) {

	//just use 7 of 9 parameters
	int ah = params[1];
	int aw = params[2];
	int range = params[9]-1;

	bool whether_local_corr = params[8];

	int ax = blockIdx.x*blockDim.x + threadIdx.x;
	int ay = blockIdx.y*blockDim.y + threadIdx.y;

	if (ax < aw && ay < ah) {

		if (whether_local_corr)
		{
			int bx = local_cor_map[0 * ah*aw + ay*aw + ax];
			int by = local_cor_map[1 * ah*aw + ay*aw + ax];

			if (bx>=0&&by>=0)
			{

				unsigned int vp = ann[ay*aw + ax];
				int xp = INT_TO_X(vp);
				int yp = INT_TO_Y(yp);
				if (xp < bx - range)
				{
					xp = bx - range;
				}

				if (xp > bx + range)
				{
					xp = bx + range;
				}

				if (yp < by - range)
				{
					yp = by - range;
				}

				if (yp > by + range)
				{
					yp = by + range;
				}

				ann[ay*aw + ax] = XY_TO_INT(xp, yp);

			}

		}

	}
}


__global__ void upSample_kernel(unsigned int * ann, unsigned int * ann_tmp,int * params, int aw_half,int ah_half) {

	int ax = blockIdx.x*blockDim.x + threadIdx.x;
	int ay = blockIdx.y*blockDim.y + threadIdx.y;

	
	int ah = params[1];
	int aw = params[2];
	int bh = params[3];
	int bw = params[4];
	
	
	float aw_ratio = (float)aw / (float)aw_half;
	float ah_ratio = (float)ah / (float)ah_half;
	int ax_half = (ax+0.5) / aw_ratio;
	int ay_half = (ay+0.5) / ah_ratio;
	ax_half = clamp(ax_half, aw_half - 1, 0);
	ay_half = clamp(ay_half, ah_half - 1, 0);
	

	if (ax < aw&&ay < ah) {

		unsigned int v_half = ann[ay_half*aw_half + ax_half];
		int bx_half = INT_TO_X(v_half);
		int by_half = INT_TO_Y(v_half);

		int bx = ax + (bx_half - ax_half)*aw_ratio + 0.5;
		int by = ay + (by_half - ay_half)*ah_ratio + 0.5;

		bx = clamp(bx, bw-1, 0);
		by = clamp(by, bh-1, 0);

		ann_tmp[ay*aw + ax] = XY_TO_INT(bx, by);
	}

}








// ********** VOTE ***********

__global__ void center_vote(unsigned int * ann, double * pb, double * pc, int * params) {//pc is for recon

	int ax = blockIdx.x*blockDim.x + threadIdx.x;
	int ay = blockIdx.y*blockDim.y + threadIdx.y;

	int ch = params[0];
	int ah = params[1];
	int aw = params[2];
	int bh = params[3];
	int bw = params[4];

	int slice_a = ah * aw;
	int pitch_a = aw;
	int slice_b = bh * bw;
	int pitch_b = bw;

	if (ax < aw&&ay < ah)
	{

		unsigned int vp = ann[ay*aw + ax];
		int xp = INT_TO_X(vp);
		int yp = INT_TO_Y(vp);
		if (yp < bh && xp < bw)
		{
			for (int i = 0; i < ch; i++)
			{

				pc[i*slice_a + ay*pitch_a + ax] = pb[i*slice_b + yp*pitch_b + xp];
			}
		}
	}
}


__global__ void center_vote(unsigned int * ann, float * pb, float * pc, int * params) {//pc is for recon

	int ax = blockIdx.x*blockDim.x + threadIdx.x;
	int ay = blockIdx.y*blockDim.y + threadIdx.y;

	int ch = params[0];
	int ah = params[1];
	int aw = params[2];
	int bh = params[3];
	int bw = params[4];

	int slice_a = ah * aw;
	int pitch_a = aw;
	int slice_b = bh * bw;
	int pitch_b = bw;

	if (ax < aw&&ay < ah)
	{

		unsigned int vp = ann[ay*aw + ax];
		int xp = INT_TO_X(vp);
		int yp = INT_TO_Y(vp);
		if (yp < bh && xp < bw)
		{
			for (int i = 0; i < ch; i++)
			{

				pc[i*slice_a + ay*pitch_a + ax] = pb[i*slice_b + yp*pitch_b + xp];
			}
		}
	}
}


__global__ void avg_vote(unsigned int * ann, float * pb, float * pc, int * params) {

	int ax = blockIdx.x*blockDim.x + threadIdx.x;
	int ay = blockIdx.y*blockDim.y + threadIdx.y;

	int ch = params[0];
	int ah = params[1];
	int aw = params[2];
	int bh = params[3];
	int bw = params[4];
	int patch_w = params[5];

	int slice_a = ah * aw;
	int pitch_a = aw;
	int slice_b = bh * bw;
	int pitch_b = bw;

	int count = 0;

	if (ax < aw&&ay < ah)
	{

		//set zero for all the channels at (ax,ay)
		for (int i = 0; i < ch; i++)
		{
			pc[i*slice_a + ay*pitch_a + ax] = 0;

		}

		//count the sum of all the possible value of (ax,ay)
		for (int dx = -patch_w / 2; dx <= patch_w / 2; dx++) {
			for (int dy = -patch_w / 2; dy <= patch_w / 2; dy++)
			{

				if ((ax + dx) < aw && (ax + dx) >= 0 && (ay + dy) < ah && (ay + dy) >= 0)
				{
					unsigned int vp = ann[(ay + dy)*aw + ax + dx];
					
					int xp = INT_TO_X(vp);
					int yp = INT_TO_Y(vp);

					if ((xp - dx) < bw && (xp - dx) >= 0 && (yp - dy) < bh && (yp - dy) >= 0)
					{
						count++;
						for (int dc = 0; dc < ch; dc++)
						{
							pc[dc*slice_a + ay*pitch_a + ax] += pb[dc*slice_b + (yp - dy)*pitch_b + xp - dx];
						}
					}
				}

			}
		}

		//count average value
		for (int i = 0; i < ch; i++)
		{
			pc[i*slice_a + ay*pitch_a + ax] /= count;
		}

	}
}













void norm(float* &dst, float* src, int channel, int height, int width){

	int count = channel*height*width;
	float* x = src;
	float* x2;
	cudaMalloc(&x2, count*sizeof(float));
	caffe_gpu_mul(count, x, x, x2);

	//caculate dis
	float*sum;
	float* ones;
	cudaMalloc(&sum, height*width*sizeof(float));
	cudaMalloc(&ones, channel*sizeof(float));
	caffe_gpu_set(channel, 1.0f, ones);
	caffe_gpu_gemv(CblasTrans, channel, height*width, 1.0f, x2, ones, 0.0f, sum);

	float *dis;
	cudaMalloc(&dis, height*width*sizeof(float));
	caffe_gpu_powx(height*width, sum, 0.5f, dis);

	//norm	
	caffe_gpu_gemm(CblasNoTrans, CblasNoTrans, channel, width*height, 1, 1.0f, ones, dis, 0.0f, x2);
	caffe_gpu_div(count, src, x2, dst);

	cudaFree(x2);
	cudaFree(ones);
	cudaFree(dis);
	cudaFree(sum);
}


__global__ void blend_cont(float *a, float *ori, int tota, float weight)
{
	int ax = blockIdx.x*blockDim.x + threadIdx.x;
	if (ax < tota)
	{

		a[ax] = ori[ax] * weight + a[ax] * (1.0 - weight);
	}
}

void blend_content(float *a, float *ori, int heighta, int widtha, int channela, float weight)
{
	dim3 blocksPerGridAB(heighta*widtha*channela / 400, 1, 1);
	dim3 threadsPerBlockAB(400, 1, 1);
	blend_cont << <blocksPerGridAB, threadsPerBlockAB >> >(a, ori, heighta*widtha*channela, weight);
}



__device__ void improve_guess(float * a, float * b, int channels, int a_rows, int a_cols, int b_rows, int b_cols, int ax, int ay, int &xbest, int &ybest, float &dbest, int xp, int yp, int patch_w, float rr, int *usageb, float maximumusage,float lambda) {
	float d, x_diff, y_diff;
	//d = dist(a, b, channels, a_rows, a_cols, b_rows, b_cols, ax, ay, xp, yp, patch_w,usageb,maximumusage);
	d = dist(a, b, channels, a_rows, a_cols, b_rows, b_cols, ax, ay, xp, yp, patch_w);
	float usage = 0, count = 0, usage1 = 0, count1 = 0;

	for (int dy = -patch_w / 2; dy <= patch_w / 2; dy++) {
		for (int dx = -patch_w / 2; dx <= patch_w / 2; dx++) {
			int by = yp, bx = xp;
			if (
				(ay + dy) < a_rows && (ay + dy) >= 0 && (ax + dx) < a_cols && (ax + dx) >= 0
				&&
				(by + dy) < b_rows && (by + dy) >= 0 && (bx + dx) < b_cols && (bx + dx) >= 0
				)//the pixel in a should exist and pixel in b should exist
			{

				usage += usageb[(by + dy) * b_cols + (bx + dx)];
				count++;
			}
		}
	}
	usage /= count;
	usage /= maximumusage;
	for (int dy = -patch_w / 2; dy <= patch_w / 2; dy++) {
		for (int dx = -patch_w / 2; dx <= patch_w / 2; dx++) {
			int by = ybest, bx = xbest;
			if (
				(ay + dy) < a_rows && (ay + dy) >= 0 && (ax + dx) < a_cols && (ax + dx) >= 0
				&&
				(by + dy) < b_rows && (by + dy) >= 0 && (bx + dx) < b_cols && (bx + dx) >= 0
				)//the pixel in a should exist and pixel in b should exist
			{

				usage1 += usageb[(by + dy) * b_cols + (bx + dx)];
				count1++;
			}
		}
	}
	usage1 /= count1;
	usage1 /= maximumusage;
	if (d + lambda*usage + rr < dbest + lambda*usage1) {
		for (int dy = -patch_w / 2; dy <= patch_w / 2; dy++) {
			for (int dx = -patch_w / 2; dx <= patch_w / 2; dx++) {
				int by = yp, bx = xp;
				if (
					(ay + dy) < a_rows && (ay + dy) >= 0 && (ax + dx) < a_cols && (ax + dx) >= 0
					&&
					(by + dy) < b_rows && (by + dy) >= 0 && (bx + dx) < b_cols && (bx + dx) >= 0
					)//the pixel in a should exist and pixel in b should exist
				{

					atomicAdd(&(usageb[(by + dy) * b_cols + (bx + dx)]), 1);

				}
			}
		}
		for (int dy = -patch_w / 2; dy <= patch_w / 2; dy++) {
			for (int dx = -patch_w / 2; dx <= patch_w / 2; dx++) {
				int by = ybest, bx = xbest;
				if (
					(ay + dy) < a_rows && (ay + dy) >= 0 && (ax + dx) < a_cols && (ax + dx) >= 0
					&&
					(by + dy) < b_rows && (by + dy) >= 0 && (bx + dx) < b_cols && (bx + dx) >= 0
					)//the pixel in a should exist and pixel in b should exist
				{

					atomicAdd(&(usageb[(by + dy) * b_cols + (bx + dx)]), -1);

				}
			}
		}
		xbest = xp;
		ybest = yp;
		dbest = d;
	}
	//}
}

__global__ void patchmatch_usage(float * a, float* b, unsigned int *ann, float *annd, int * params, int *usageb, float maximumusage,float lambda) {

	int ax = blockIdx.x*blockDim.x + threadIdx.x;
	int ay = blockIdx.y*blockDim.y + threadIdx.y;

	//assign params
	int ch = params[0];
	int a_rows = params[1];
	int a_cols = params[2];
	int b_rows = params[3];
	int b_cols = params[4];
	int patch_w = params[5];
	int pm_iters = params[6];
	int rs_max = params[7];
	if (ax < a_cols && ay < a_rows) {

		// for random number
		curandState state;
		InitcuRand(state);

		unsigned int v, vp;

		int xp, yp, xbest, ybest;

		int xmin, xmax, ymin, ymax;

		float dbest;
		v = ann[ay*a_cols + ax];
		xbest = INT_TO_X(v), ybest = INT_TO_Y(v);
		annd[ay*a_cols + ax] = dist(a, b, ch, a_rows, a_cols, b_rows, b_cols, ax, ay, xbest, ybest, patch_w);

		for (int iter = 0; iter < pm_iters; iter++) {

			/* Current (best) guess. */
			v = ann[ay*a_cols + ax];
			xbest = INT_TO_X(v), ybest = INT_TO_Y(v);
			dbest = annd[ay*a_cols + ax];
			/* In each iteration, improve the NNF, by jumping flooding. */
			for (int jump = 8; jump > 0; jump /= 2) {

				/* Propagation: Improve current guess by trying instead correspondences from left, right, up and downs. */
				if ((ax - jump) < a_cols && (ax - jump) >= 0)//left
				{
					vp = ann[ay*a_cols + ax - jump];//the pixel coordinates in image b

					xp = INT_TO_X(vp) + jump, yp = INT_TO_Y(vp);//the propagated match from vp, the center of the patch, which should be in the image

					if (yp >= 0 && yp < b_rows && xp >= 0 && xp < b_cols)
					{
						//improve guess
						improve_guess(a, b, ch, a_rows, a_cols, b_rows, b_cols, ax, ay, xbest, ybest, dbest, xp, yp, patch_w, 0., usageb, maximumusage,lambda);
						ann[ay*a_cols + ax] = XY_TO_INT(xbest, ybest);
						annd[ay*a_cols + ax] = dbest;
					}
				}

				if ((ax + jump) < a_cols)//right
				{
					vp = ann[ay*a_cols + ax + jump];//the pixel coordinates in image b

					xp = INT_TO_X(vp) - jump, yp = INT_TO_Y(vp);

					if (yp >= 0 && yp < b_rows && xp >= 0 && xp < b_cols)
					{
						//improve guess
						improve_guess(a, b, ch, a_rows, a_cols, b_rows, b_cols, ax, ay, xbest, ybest, dbest, xp, yp, patch_w, 0., usageb, maximumusage, lambda);
						ann[ay*a_cols + ax] = XY_TO_INT(xbest, ybest);
						annd[ay*a_cols + ax] = dbest;
					}
				}

				if ((ay - jump) < a_rows && (ay - jump) >= 0)//up
				{
					vp = ann[(ay - jump)*a_cols + ax];//the pixel coordinates in image b
					xp = INT_TO_X(vp), yp = INT_TO_Y(vp) + jump;

					if (yp >= 0 && yp < b_rows && xp >= 0 && xp < b_cols)
					{

						//improve guess
						improve_guess(a, b, ch, a_rows, a_cols, b_rows, b_cols, ax, ay, xbest, ybest, dbest, xp, yp, patch_w, 0., usageb, maximumusage, lambda);
						ann[ay*a_cols + ax] = XY_TO_INT(xbest, ybest);
						annd[ay*a_cols + ax] = dbest;
					}
				}

				if ((ay + jump) < a_rows)//down
				{
					vp = ann[(ay + jump)*a_cols + ax];//the pixel coordinates in image b	
					xp = INT_TO_X(vp), yp = INT_TO_Y(vp) - jump;

					if (yp >= 0 && yp < b_rows && xp >= 0 && xp < b_cols)
					{
						//improve guess
						improve_guess(a, b, ch, a_rows, a_cols, b_rows, b_cols, ax, ay, xbest, ybest, dbest, xp, yp, patch_w, 0., usageb, maximumusage, lambda);
						ann[ay*a_cols + ax] = XY_TO_INT(xbest, ybest);
						annd[ay*a_cols + ax] = dbest;
					}
				}

			}

			/* Random search: Improve current guess by searching in boxes of exponentially decreasing size around the current best guess. */
			int rs_start = rs_max;
			if (rs_start > cuMax(b_cols, b_rows)) {
				rs_start = cuMax(b_cols, b_rows);
			}
			for (int mag = rs_start; mag >= 1; mag /= 2) {
				/* Sampling window */
				xmin = cuMax(xbest - mag, 0), xmax = cuMin(xbest + mag + 1, b_cols);
				ymin = cuMax(ybest - mag, 0), ymax = cuMin(ybest + mag + 1, b_rows);
				xp = xmin + (int)(MycuRand(state)*(xmax - xmin)) % (xmax - xmin);
				yp = ymin + (int)(MycuRand(state)*(ymax - ymin)) % (ymax - ymin);

				//improve guess
				float usage = 0, count = 0;
				for (int dy = -patch_w / 2; dy <= patch_w / 2; dy++) {
					for (int dx = -patch_w / 2; dx <= patch_w / 2; dx++) {
						int by = yp, bx = xp;
						if (
							(ay + dy) < a_rows && (ay + dy) >= 0 && (ax + dx) < a_cols && (ax + dx) >= 0
							&&
							(by + dy) < b_rows && (by + dy) >= 0 && (bx + dx) < b_cols && (bx + dx) >= 0
							)//the pixel in a should exist and pixel in b should exist
						{

							usage += usageb[(by + dy) * b_cols + (bx + dx)];
							count++;
						}
					}
				}
				usage /= count;
				float usage1 = 0, count1 = 0;
				for (int dy = -patch_w / 2; dy <= patch_w / 2; dy++) {
					for (int dx = -patch_w / 2; dx <= patch_w / 2; dx++) {
						int by = ybest, bx = xbest;
						if (
							(ay + dy) < a_rows && (ay + dy) >= 0 && (ax + dx) < a_cols && (ax + dx) >= 0
							&&
							(by + dy) < b_rows && (by + dy) >= 0 && (bx + dx) < b_cols && (bx + dx) >= 0
							)//the pixel in a should exist and pixel in b should exist
						{

							usage1 += usageb[(by + dy) * b_cols + (bx + dx)];
							count1++;
						}
					}
				}
				usage1 /= count1;
				if (usage <= maximumusage * 2 || usage<usage1)
					improve_guess(a, b, ch, a_rows, a_cols, b_rows, b_cols, ax, ay, xbest, ybest, dbest, xp, yp, patch_w, FLT_MIN, usageb, maximumusage, lambda);

			}

			ann[ay*a_cols + ax] = XY_TO_INT(xbest, ybest);
			annd[ay*a_cols + ax] = dbest;
			__syncthreads();
		}
	}
}

__global__ void initialAnn_kernel(unsigned int * ann, int * params, float *local_cor_map,int*usageb,int patch_w) {
	int ah = params[1];
	int aw = params[2];
	int bh = params[3];
	int bw = params[4];

	int ax = blockIdx.x*blockDim.x + threadIdx.x;
	int ay = blockIdx.y*blockDim.y + threadIdx.y;

	curandState state;
	InitcuRand(state);

	if (ax < aw && ay < ah) {
		int bx = (int)(MycuRand(state)*bw);
		int by = (int)(MycuRand(state)*bh);
		ann[ay*aw + ax] = XY_TO_INT(bx, by);

	}
}

__global__ void usage_count(unsigned int *ann, int * params, int *usageb)
{
	int ax = blockIdx.x*blockDim.x + threadIdx.x;
	int ay = blockIdx.y*blockDim.y + threadIdx.y;

	//assign params
	int ch = params[0];
	int ah = params[1];
	int aw = params[2];
	int bh = params[3];
	int bw = params[4];
	int patch_w = params[5];
	if (ay < ah && ax < aw)
	{
		int by = INT_TO_Y(ann[ay*aw + ax]);
		int bx = INT_TO_X(ann[ay*aw + ax]);
		for (int dy = -patch_w / 2; dy <= patch_w / 2; dy++) {
			for (int dx = -patch_w / 2; dx <= patch_w / 2; dx++) {
				if (
					(ay + dy) < ah && (ay + dy) >= 0 && (ax + dx) < aw && (ax + dx) >= 0
					&&
					(by + dy) < bh && (by + dy) >= 0 && (bx + dx) < bw && (bx + dx) >= 0
					)
				{

					atomicAdd(&(usageb[(by + dy) * bw + (bx + dx)]), 1);

				}
			}

		}
	}
}



void match_usage(float *device_dataa, float *device_datab, int channela, int heighta, int widtha, int heightb, int widthb, int patchsize, unsigned* &ann, int ah_half, int aw_half,int tt,float alpha, float lambda)
{
	dim3 blocksPerGridAB(widtha / 20 + 1, heighta / 20 + 1, 1);
	dim3 threadsPerBlockAB(20, 20, 1);
	int params[10];
	params[0] = channela;
	params[1] = heighta;
	params[2] = widtha;
	params[3] = heightb;
	params[4] = widthb;
	params[5] = patchsize;
	params[6] = 11;
	if (ann == NULL)
	{
		params[7] = heighta > widtha ? heighta/16 : widtha/16;
	}
	else params[7] = 6;
	float maximumusage = heighta*widtha;
	maximumusage /= heightb*widthb;
	maximumusage *= patchsize*patchsize;
	int *params_device;
	cudaMalloc(&params_device, 10 * sizeof(int));
	cudaMemcpy(params_device, params, 10 * sizeof(int), cudaMemcpyHostToDevice);
	int tmpa = heighta*widtha;
	int tmpb = heightb*widthb;
	float *data_a_ori;
	cudaMalloc(&data_a_ori, heighta*widtha*channela*sizeof(float));
	cudaMemcpy(data_a_ori, device_dataa, heighta*channela*widtha*sizeof(float), cudaMemcpyDeviceToDevice);
	unsigned *ann_device_AB;
	float *annd_device_AB;
	int *usageb;

	cudaMalloc(&ann_device_AB, tmpa*sizeof(unsigned));
	cudaMalloc(&annd_device_AB, tmpa*sizeof(float));
	cudaMalloc(&usageb, tmpb*sizeof(int));
	
	float *data_a_N, *data_b_N;
	cudaMalloc(&data_a_N, tmpa*channela*sizeof(float));
	cudaMalloc(&data_b_N, tmpb*channela*sizeof(float));
	if (ann == NULL)
		initialAnn_kernel << <blocksPerGridAB, threadsPerBlockAB >> >(ann_device_AB, params_device, NULL, usageb, patchsize);
	else
		upSample_kernel << <blocksPerGridAB, threadsPerBlockAB >> >(ann, ann_device_AB, params_device, aw_half, ah_half);
	cudaMemset(usageb, 0, tmpb*sizeof(int));
	usage_count << <blocksPerGridAB, threadsPerBlockAB >> >(ann_device_AB, params_device, usageb);
	norm(data_b_N, device_datab, channela, heightb, widthb);
	if (tt != 1)
	{
		for (int turn = 0; turn < 5; turn++)
		{
			norm(data_a_N, device_dataa, channela, heighta, widtha);
			patchmatch_usage << <blocksPerGridAB, threadsPerBlockAB >> >(data_a_N, data_b_N, ann_device_AB, annd_device_AB, params_device, usageb, maximumusage,lambda);
			avg_vote << <blocksPerGridAB, threadsPerBlockAB >> >(ann_device_AB, device_datab, device_dataa, params_device);
			blend_content(device_dataa, data_a_ori, heighta, widtha, channela, alpha);
		}
	}
	else
	{
		norm(data_a_N, device_dataa, channela, heighta, widtha);
		patchmatch_usage << <blocksPerGridAB, threadsPerBlockAB >> >(data_a_N, data_b_N, ann_device_AB, annd_device_AB, params_device, usageb, maximumusage,lambda);
		avg_vote << <blocksPerGridAB, threadsPerBlockAB >> >(ann_device_AB, device_datab, device_dataa, params_device);
		blend_content(device_dataa, data_a_ori, heighta, widtha, channela, 0.5f);
	}

	if (ann != NULL) cudaFree(ann);
	cudaMalloc(&ann, tmpa*sizeof(unsigned));
	cudaMemcpy(ann, ann_device_AB, tmpa*sizeof(unsigned), cudaMemcpyDeviceToDevice);
	cudaFree(ann_device_AB);
	cudaFree(annd_device_AB);
	cudaFree(params_device);
	cudaFree(data_a_N);
	cudaFree(data_b_N);

}

