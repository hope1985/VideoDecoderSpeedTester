

#include "kernel.cuh"


__global__ void change_brightness(uint8_t* d_y, int width, int height, int pitch, int brightness) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        uint8_t* row = d_y + y * pitch;
        row[x] = min(255, row[x] + brightness); // brighten pixel
    }
}

__global__ void invert_luma(uint8_t* y_plane, int width, int height, int pitch) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x < width && y < height) {
        y_plane[y * pitch + x] = 255 - y_plane[y * pitch + x];
    }
}

void call_cuda_kernel(uint8_t* d_y_plane,int width,int height, int pitch)
{
 
    dim3 block(32, 32);
    dim3 grid((width + 31) / 32, (height + 31) / 32);

    invert_luma << <grid, block >> > (d_y_plane, width, height, pitch);

	cudaDeviceSynchronize(); // Ensure the kernel execution is complete
}