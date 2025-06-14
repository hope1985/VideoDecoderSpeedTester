

#include "dtype_conversion.cuh"

/*
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
}*/


//====================================================
template<class T>
__global__ void nv12_10bit_yuv420_10bit_le(
    const uint16_t* y_src,
    const uint16_t* uv_src,
    int y_pitch, int uv_pitch,
    T* y_out,
    T* u_out,
    T* v_out,
    int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;


//############## Each 16-bit little-endian sample has valid data in upper 10 bits ##############
//We need to shift right by 6 bits to convert to 8-bit YUV42010P format

    // Copy Y plane (full resolution)
    if (x < width && y < height) {
        const T* y_row = (const uint16_t*)((const uint8_t*)y_src + y * y_pitch);
        y_out[y * width + x] = y_row[x]>>6;
    }

    // Copy UV plane (subsampled by 2)
    if (x < width / 2 && y < height / 2) {
        const T* uv_row = (const uint16_t*)((const uint8_t*)uv_src + y * uv_pitch);
        int idx = y * (width / 2) + x;

        u_out[idx] = uv_row[2 * x]>>6;     // U
        v_out[idx] = uv_row[2 * x + 1]>>6; // V
    }
}

//AV_PIX_FMT_P010LE to AV_PIX_FMT_YUV420P10LE
void convert_nv12_10bit_yuv420_10bit_le(AVFrame* frame, uint8_t* d_yf, uint8_t* d_uf, uint8_t* d_vf)
{

    dim3 block(16, 16);
    dim3 grid((frame->width + 15) / 16, (frame->height + 15) / 16);

    // AVFrame* frame is in GPU memory (AV_PIX_FMT_CUDA / NV12)
    nv12_10bit_yuv420_10bit_le << <grid, block >> > (
        (uint16_t*)frame->data[0], 
        (uint16_t*)frame->data[1], frame->linesize[0], frame->linesize[1],
        (uint16_t*)d_yf, (uint16_t*)d_uf, (uint16_t*)d_vf,
        frame->width, frame->height);


    cudaDeviceSynchronize(); // Make sure the kernel finishes
}

template<class T>
__global__ void nv12_to_yuv420_8bit_kernel(
    const uint8_t* y_src,
    const uint8_t* uv_src,
    int y_pitch, int uv_pitch,
    T* y_out,
    T* u_out,
    T* v_out,
    int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;


    // Copy Y plane (full resolution)
    if (x < width && y < height) {
        const T* y_row = (const T*)(y_src + y * y_pitch);
        y_out[y * width + x] = y_row[x];
    }

    // Copy UV plane (subsampled by 2)
    if (x < width / 2 && y < height / 2) {
        const T* uv_row = (const T*)(uv_src + y * uv_pitch);
        int idx = y * (width / 2) + x;

        u_out[idx] = uv_row[2 * x];     // U
        v_out[idx] = uv_row[2 * x + 1] ; // V
    }
}

//AV_PIX_FMT_P010LE to AV_PIX_FMT_YUV420P10LE
void convert_nv12_8bit_yuv420_8bit(AVFrame* frame, uint8_t* d_yf, uint8_t* d_uf, uint8_t* d_vf)
{
    
    dim3 block(16, 16);
    dim3 grid((frame->width + 15) / 16, (frame->height + 15) / 16);

    // AVFrame* frame is in GPU memory (AV_PIX_FMT_CUDA / NV12)
    nv12_to_yuv420_8bit_kernel << <grid, block >> > (
        frame->data[0],
        frame->data[1], 
        frame->linesize[0], frame->linesize[1],
        d_yf, d_uf, d_vf,
        frame->width, frame->height);

    cudaDeviceSynchronize(); // Make sure the kernel finishes
}


void init_cuda_memory(void** buf,   int w,  int h,int pixel_size)
{
    //cudaError_t cudaStatus = cudaMallocHost((void**)&buf, w * h * pixel_size);  //pinned
    cudaError_t cudaStatus=cudaMalloc((void**)&(*buf), w*h * pixel_size);
    if (cudaStatus != cudaSuccess) {
        std::cerr << "cudaMalloc failed!" << std::endl;
    }
}

void free_cuda_memory(void** buf)
{
    cudaFree(*buf);

}