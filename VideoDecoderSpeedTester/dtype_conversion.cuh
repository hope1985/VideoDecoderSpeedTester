#include "cuda_runtime.h"
extern "C" {
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libavutil/hwcontext.h>
#include <libavutil/imgutils.h>
#include <libswscale/swscale.h>
}

#include <iostream>
//__global__ void change_brightness(uint8_t* d_y, int width, int height, int pitch);
//__global__ void invert_luma(uint8_t* d_y, int width, int height, int pitch);

void convert_nv12_8bit_yuv420_8bit(AVFrame* frame, uint8_t* d_yf, uint8_t* d_uf, uint8_t* d_vf);
void convert_nv12_10bit_yuv420_10bit_le(AVFrame* frame, uint8_t* d_yf, uint8_t* d_uf, uint8_t* d_vf);

void init_cuda_memory(void** buf,   int w,  int h, int pixel_size);
void free_cuda_memory(void** buf);