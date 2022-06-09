#include <iostream>
#include <vector>
#include <stdlib.h>

#include "hip/hip_runtime_api.h"
#include "rocblas.h"
#include "error.hpp"

using namespace std;

__host__ void saxpy(int n) {
    float *x = nullptr;
    float *y = nullptr;
    float *d_x = nullptr;
    float *d_y = nullptr;
    int N = n * n;

    rocblas_initialize();
    //hipStream_t stream;
    //if (hipStreamCreate(&stream) != hipSuccess) {
    //    return EXIT_FAILURE;
    //}
    rocblas_handle handle;
    if (rocblas_create_handle(&handle) != rocblas_status_success) {
        cout << "rocblas create handle failure" << endl;
        exit(-1);
    }
    //if (rocblas_set_stream(handle, stream) != rocblas_status_success) {
    //    return EXIT_FAILURE;
    //}
    hipError_t err;
    x = reinterpret_cast<float *> (malloc(sizeof(float) * 2 * N));
    y = x + N;
    err = hipMalloc(reinterpret_cast<void**>(&d_x), 2 * N * sizeof(float));

    if (err != hipSuccess) {
        cout << "hip malloc failure" << endl;
    }
    d_y = d_x + N;

    for (int ii = 0; ii < 2 * N; ii++) {
        x[ii] = rand() / RAND_MAX;
    }

    rocblas_set_vector(N, sizeof(float), x, 1, d_x, 1);
    rocblas_set_vector(N, sizeof(float), y, 1, d_y, 1);

    float alpha = 2.0f;
    int nIter = 5000;
    hipEvent_t start, stop;
    hipEventCreate(&start);
    hipEventCreate(&stop);

    hipEventRecord(start);
    for (int ii = 0; ii < nIter; ++ii) {
        rocblas_saxpy(handle, N, &alpha, d_x, 1, d_y, 1);
    }
    hipEventRecord(stop);
    rocblas_get_vector(N, sizeof(float), d_y, 1, y, 1);
    hipDeviceSynchronize();
    HIPErrCheck(hipGetLastError());
    
    float millisecons = 0.0f;
    hipEventElapsedTime(&millisecons, start, stop);
    double avg_time = millisecons / nIter;
    double GFlops = 1e-6 * N * 2 / avg_time;
    double GBs = 1e-6 * N *sizeof(float) * 3 / avg_time;
    printf("%dx%d, %.3lf ms, %.3lf GFLOPS, %.3lf GB/s\n", n, n, avg_time, GFlops, GBs);

    //if (hipstreamDestroy(stream) != hipSuccess) {
    //    return EXIT_FAILURE;
    //}
    if (rocblas_destroy_handle(handle) != rocblas_status_success) {
        cout << "rocblas create handle failure" << endl;
    }
    hipFree(d_x);
    free(x);
}

int main() {
    int n = 4096;
    saxpy(n);
    return 0;
}
