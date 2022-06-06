#include <assert.h>

#define HIPErrcheck(res) { HIPAssert((res), __FILE__, __LINE__); }
inline void HIPAssert(hipError_t code, const char *file, int line, bool abort=true) {
    if (code != hipSuccess) {
        fprintf(stderr,"hip assert: %s %s %d\n", hipGetErrorString(code), file, line);
    }
}

