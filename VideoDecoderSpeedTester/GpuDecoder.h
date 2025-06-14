#ifndef GPUDECODER_H
#define GPUDECODER_H

extern "C" {
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libavutil/hwcontext.h>
#include <libavutil/imgutils.h>
#include <libswscale/swscale.h>
}
#define DTYPE uint16_t
#include <string>
#include <iostream>
#include "dtype_conversion.cuh"
class GpuDecoder {
public:
    GpuDecoder();
    ~GpuDecoder();
	//FILE* fp = nullptr;
    void* devY;
    void* devU;
    void* devV;
    int id;
    bool open(int decoderID,const std::string& filepath, int start_frame_index = 0, int num_frames=0);
    bool decode_next_frame(AVFrame* output_yuv420p_frame);
    void close();
    AVCodecContext* codec_ctx_ = nullptr;
private:
    static enum AVPixelFormat get_hw_format(AVCodecContext* ctx, const enum AVPixelFormat* pix_fmts);

    AVFormatContext* fmt_ctx_ = nullptr;
    //AVCodecContext* codec_ctx_ = nullptr;
    AVStream* video_stream_ = nullptr;
    const AVCodec* decoder_ = nullptr;
    AVFrame* mapped_frame;
    AVPacket* pkt_ = nullptr;
    AVFrame* frame_ = nullptr;
    AVFrame* sw_frame_ = nullptr;
    AVBufferRef* hw_device_ctx_ = nullptr;
    SwsContext* sws_ctx_ = nullptr;
    int video_stream_index_ = -1;
    int current_frame_index_ = 0;
    int start_frame_index_ = 0;
    int end_frame_index_ = INT32_MAX;
};

#endif 