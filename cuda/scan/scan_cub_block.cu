#include "scan.cuh"
using namespace cub;

template <
    int                 TPB,
    int                 IPT,
    BlockScanAlgorithm  ALGORITHM>
__global__ void BlockPrefixSumKernel_cub(float *out, float *in) {
    typedef BlockLoad<float, TPB, IPT, BLOCK_LOAD_WARP_TRANSPOSE> BlockLoadT;
    typedef BlockStore<float, TPB, IPT, BLOCK_STORE_WARP_TRANSPOSE> BlockStoreT;
    typedef BlockScan<float, TPB, ALGORITHM> BlockScanT;

    __shared__ union TempStorage {
        typename BlockLoadT::TempStorage    load;
        typename BlockStoreT::TempStorage   store;
        typename BlockScanT::TempStorage    scan;
    } temp_storage;

    float data[IPT];
    BlockLoadT(temp_storage.load).Load(in, data);
    __syncthreads();
    
    float aggregate;
    BlockScanT(temp_storage.scan).ExclusiveSum(data, data, aggregate);
    __syncthreads();

    BlockStoreT(temp_storage.store).Store(out, data);

    if (threadIdx.x == 0) out[TPB * IPT] = aggregate;
}

void init_h(float *&h, int num) {
    for(int ii = 0; ii < num; ii++) 
        h[ii] = float(ii);
}

int main(int argc, char **argv) {
    int numElements = 4096;
    if(argc > 1) numElements = std::atoi(argv[1]);
    float *d_in = nullptr;
    float *d_out = nullptr;
    float *h_in = new float [numElements];
    float *h_out = new float [numElements + 1];
        
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaMalloc(&d_in, numElements * sizeof(float));
    cudaMalloc(&d_out, (numElements + 1) * sizeof(float));
    init_h(h_in, numElements);
    cuErrCheck(cudaMemcpy(d_in, h_in, sizeof(float) * numElements, cudaMemcpyHostToDevice));
    
    cudaEventRecord(start);
    if (numElements == 256) {
        BlockPrefixSumKernel_cub<TPB64, 4, BLOCK_SCAN_RAKING><<<1, TPB64>>>(
            d_out,
            d_in);
    }
    else if (numElements == 1024) {
        BlockPrefixSumKernel_cub<TPB256, 4, BLOCK_SCAN_RAKING><<<1, TPB256>>>(
            d_out,
            d_in);
    }
    else if (numElements == 4096) {
        BlockPrefixSumKernel_cub<TPB1024, 4, BLOCK_SCAN_RAKING><<<1, TPB1024>>>(
            d_out,
            d_in);
    }
    cuErrCheck(cudaGetLastError());

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0.0f;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Cub Scan %i elements takes %f ms\n", numElements, milliseconds);
    cuErrCheck(cudaMemcpy(h_out, d_out, sizeof(float) * (numElements + 1), cudaMemcpyDeviceToHost));
    printf("Result  :  sum = %f\n", h_out[numElements-1]);

    cudaFree(d_in);
    cudaFree(d_out);
    delete[] h_in;
    delete[] h_out;
    return 0;
}
