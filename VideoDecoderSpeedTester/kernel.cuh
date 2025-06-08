#include "cuda_runtime.h"

__global__ void change_brightness(uint8_t* d_y, int width, int height, int pitch);
__global__ void invert_luma(uint8_t* d_y, int width, int height, int pitch);
void call_cuda_kernel(uint8_t* d_y_plane, int width, int height, int pitch);