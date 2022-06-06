#include <hip/hip_runtime.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include "color.hpp"
#include "error.hpp"
using namespace std;

__global__ void add(double *a, double *b, double *c, int n) {
    int idx = hipThreadIdx_x + hipBlockDim_x * hipBlockIdx_x;
    c[idx] = a[idx] + b[idx];
}

void InitHost(double *va, double *vb, double *ref, int n) {
    for (int ii = 0; ii < n; ii++) {
        va[ii] = ii;
        vb[ii] = n - ii;
        ref[ii] = n;
    }
}

void Check(double *out, double *ref, int n) {
    for (int ii = 0; ii < n; ii++) {
        assert(out[ii] == ref[ii]);
    }
    Color::Modifier green(Color::FG_GREEN);
    Color::Modifier def(Color::FG_DEFAULT);
    cout << green <<"Pass" << def << endl;
}

int main(int argc, char *argv[]) {
    int n = 2048;
    double *h_va = reinterpret_cast<double *>(malloc(n * sizeof(double)));
    double *h_vb = reinterpret_cast<double *>(malloc(n * sizeof(double)));
    double *h_out = reinterpret_cast<double *>(malloc(n * sizeof(double)));
    double *h_ref = reinterpret_cast<double *>(malloc(n * sizeof(double)));
    InitHost(h_va, h_vb, h_ref, n);

    double *d_va;
    double *d_vb;
    double *d_out;
    hipMalloc(reinterpret_cast<void **>(&d_va), n * sizeof(double));
    hipMalloc(reinterpret_cast<void **>(&d_vb), n * sizeof(double));
    hipMalloc(reinterpret_cast<void **>(&d_out), n * sizeof(double));

    hipMemcpy(d_va, h_va, n * sizeof(double), hipMemcpyHostToDevice);
    hipMemcpy(d_vb, h_vb, n * sizeof(double), hipMemcpyHostToDevice);
    
    dim3 TPB(128);
    dim3 BPG((n + 127) / 128);

    hipLaunchKernelGGL(add, BPG, TPB, 0, 0, d_va, d_vb, d_out, n);
    HIPErrcheck(hipGetLastError());
    
    hipMemcpy(h_out, d_out, sizeof(double) * n, hipMemcpyDeviceToHost);
    hipFree(d_va);
    hipFree(d_vb);
    hipFree(d_out);

    Check(h_out, h_ref, n);
    free(h_va);
    free(h_vb);
    free(h_ref);
    free(h_out);

    return 0;
}
