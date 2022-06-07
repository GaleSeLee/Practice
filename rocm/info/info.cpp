#include <hip/hip_runtime.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
using namespace std;

int main(int argc, char *argv[]) {
    hipDeviceProp_t prop;
    int count;
    hipGetDeviceCount(&count);

    for(int ii = 0; ii < count; ii++) {
        hipGetDeviceProperties(&prop ,ii);
        cout << prop.name << endl;
	cout << prop.totalGlobalMem/1024/1024 << endl;
    }

    return 0;
}
