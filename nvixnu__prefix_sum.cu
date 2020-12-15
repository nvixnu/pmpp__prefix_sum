#include <math.h>
#include "nvixnu__prefix_sum.h"


__global__
void nvixnu__kogge_stone_scan_by_block_kernel(double *input, double *output, const int length, double *last_sum){
	extern __shared__ double section_sums[];

	int tid = blockIdx.x*blockDim.x + threadIdx.x;


	if(tid < length){
		section_sums[threadIdx.x] = input[tid];
	}

	unsigned int stride;
	for( stride= 1; stride < blockDim.x; stride *= 2){
		__syncthreads();
		if(threadIdx.x >= stride){
			section_sums[threadIdx.x] += section_sums[threadIdx.x - stride];
		}
	}
	output[tid] = section_sums[threadIdx.x];
	if(last_sum != NULL && threadIdx.x == (blockDim.x - 1)){
		last_sum[blockIdx.x] = section_sums[threadIdx.x];
	}
}

__global__
void nvixnu__brent_kung_scan_by_block_kernel(double *input, double *output, const int length, double *last_sum){
	extern __shared__ double section_sums[];

	int tid = blockIdx.x*blockDim.x + threadIdx.x;

	if(tid < length){
		section_sums[threadIdx.x] = input[tid];
	}

	__syncthreads();


	for(unsigned int stride = 1; stride < blockDim.x; stride *= 2){
		__syncthreads();
		int idx = (threadIdx.x + 1) * 2 * stride - 1;
		if(idx < blockDim.x){
			section_sums[idx] += section_sums[idx - stride];
		}
	}

	for(int stride = blockDim.x/4; stride > 0; stride /=2){
		__syncthreads();
		int idx = (threadIdx.x + 1) * 2 *stride - 1;
		if((idx + stride) < blockDim.x){
			section_sums[idx + stride] += section_sums[idx];
		}
	}
	__syncthreads();

	output[tid] = section_sums[threadIdx.x];
	if(last_sum != NULL && threadIdx.x == (blockDim.x - 1)){
		last_sum[blockIdx.x] = section_sums[threadIdx.x];
	}
}

__global__
void nvixnu__kogge_stone_3_phase_scan_by_block_kernel(double *input, double *output, const int length, const int section_length, double *last_sum){
	extern __shared__ double section_sums[];
	int b_dim = blockDim.x;

	// How many phases we should have in order to load the input array to shared memory in a coalesced manner (corner turning)
	int phases_count = ceil(section_length/(double)b_dim);
	// The subsection length is setted to be equals to the phases_count, in order to use all threads in the subsection scan
	int sub_section_max_length = phases_count;


	// Phase 1: Corner turning to load the input data into shared memory
	for(int i = 0; i < phases_count; i++){
		int shared_mem_index = i*b_dim + threadIdx.x;
		int input_index = blockIdx.x*section_length + shared_mem_index;
		//This comparison could be removed if we handle the last phase separately and using the dynamic blockIndex assignment
		if(input_index < length && shared_mem_index < section_length){
			section_sums[shared_mem_index] = input[input_index];
		}
	}

	__syncthreads();

	//Phase 1: Perform the scan on each sub_section
	for(int i = 1; i < sub_section_max_length; i++){
		int index = threadIdx.x*sub_section_max_length + i;
		if(index < section_length){
			section_sums[index] += section_sums[index -1];
		}
	}

	__syncthreads();


	//Phase 2: Performs the Kogge-Stone scan for the last element of each subsection. This step could be performed also by Brent-Kung scan
	for(int stride= 1; stride < section_length; stride *= 2){
		__syncthreads();
		// sub_section_length*threadIdx.x: Indicates the start position of each subsection
		// sub_section_length -1: The last item in a given subsection
		int last_element = sub_section_max_length*threadIdx.x + sub_section_max_length -1;
		if(threadIdx.x >= stride && last_element < section_length){
			section_sums[last_element] += section_sums[last_element - stride*sub_section_max_length];
		}
	}




	__syncthreads();

	//Phase 3: Adding the last element of previous sub_section
	for(int i = 0; i < sub_section_max_length - 1; i++){
		__syncthreads();
		if(threadIdx.x != 0){
			int index = threadIdx.x*sub_section_max_length + i;
			if(index < section_length){
				section_sums[index] += section_sums[threadIdx.x*sub_section_max_length - 1];
			}
		}
	}

	//Save the data on the output array
	for(int i = 0; i < phases_count; i++){
		int output_index = blockIdx.x*section_length + i*b_dim + threadIdx.x;
		if(i*b_dim + threadIdx.x < section_length){
			output[output_index] = section_sums[i*b_dim + threadIdx.x];
		}
	}

	if(last_sum != NULL && threadIdx.x == 0){
		last_sum[blockIdx.x] = section_sums[section_length - 1];
	}


}

