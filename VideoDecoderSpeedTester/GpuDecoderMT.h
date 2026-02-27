#ifndef GPUDECODERMT_H
#define GPUDECODERMT_H


extern "C" {
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libavutil/hwcontext.h>
#include <libavutil/imgutils.h>
#include <libswscale/swscale.h>
}
#define DTYPE uint16_t
#define BUFFER_SIZE 2
#include <string>
#include <iostream>
#include <vector>
#include "dtype_conversion.cuh"
#include <thread>
#include <atomic>
class GpuDecoderMT {
public:
    GpuDecoderMT();
    ~GpuDecoderMT();
    //FILE* fp = nullptr;
    void* devY;
    void* devU;
    void* devV;

    void* devY1;
    void* devU1;
    void* devV1;
    int id;
    int width;
    int height;
    bool open(int decoderID, const std::string& filepath, int start_frame_index = 0, int num_frames = 0);
    bool decode_next_frame(AVFrame* output_yuv420p_frame);
    bool decode_next_frame(int fidx);
    bool decode_next_frame();
    void close();
    AVCodecContext* codec_ctx_ = nullptr;

    std::atomic<int> read_idx{ 0 };
    std::atomic<int> status[BUFFER_SIZE]; // 0 = empty, 1 = full 
    void producer();
    FILE* fp1 = nullptr;
    FILE* fp2 = nullptr;
private:
    static enum AVPixelFormat get_hw_format(AVCodecContext* ctx, const enum AVPixelFormat* pix_fmts);

    AVFormatContext* fmt_ctx_ = nullptr;
    //AVCodecContext* codec_ctx_ = nullptr;
    AVStream* video_stream_ = nullptr;
    const AVCodec* decoder_ = nullptr;
    AVFrame* mapped_frame = nullptr;
    AVFrame* mapped_frame1 = nullptr;
    AVPacket* pkt_ = nullptr;
    AVFrame* frame_ = nullptr;
    AVFrame* frame_1 = nullptr;

    AVFrame* sw_frame_ = nullptr;
    AVBufferRef* hw_device_ctx_ = nullptr;
    SwsContext* sws_ctx_ = nullptr;
    int video_stream_index_ = -1;
    int current_frame_index_ = 0;
    int start_frame_index_ = 0;
    int end_frame_index_ = INT32_MAX;
    AVBufferRef* hw_frames_ctx;

    std::atomic<int> write_idx{ 0 };
    std::thread* prod = nullptr;
    std::atomic<bool> stop_flag{ false };
    AVFrame* frames[BUFFER_SIZE];

};

#endif 

