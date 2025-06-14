#ifndef HELPER_H
#define HELPER_H


#include <iostream>
#include <istream>
#include <fstream>
#include <string>
#include <cstring>   // needs for GCC

using namespace std;

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/hwcontext.h>
#include <libavutil/imgutils.h>
#include <libavutil/opt.h>
}


static void save_yuv420_frame(AVFrame* frame, int width, int height, const std::string& filename) {
    std::ofstream out(filename, std::ios::binary);
    for (int i = 0; i < height; i++) out.write((char*)frame->data[0] + i * frame->linesize[0], width);             // Y
    for (int i = 0; i < height / 2; i++) out.write((char*)frame->data[1] + i * frame->linesize[1], width / 2);     // U
    for (int i = 0; i < height / 2; i++) out.write((char*)frame->data[2] + i * frame->linesize[2], width / 2);     // V
    out.close();
}

// Function to open YUV420 file
static ifstream open_YUV420_file( string filepath, int W, int H, int bd, int startFrame = 0) {
    int bpp = 1; // bytes per pixel
    if (bd == 10) {
        bpp = 2;
    }

    int Ybyte = W * H * bpp;
    int UVbyte = Ybyte / 2;

    ifstream yuv_f(filepath, ios::binary);


    yuv_f.seekg((Ybyte + UVbyte)* startFrame, ios::cur);
    
    auto s = yuv_f.is_open();
    return yuv_f;
}


template<class  T>
static  bool read_YUV420_frame(ifstream& yuv_f, T* y, T* u, T* v, int width, int height) {

    
	if (yuv_f.eof()) {
		return false; // End of file reached
	}

    int Ypixels = width * height;
    int UVpixels = Ypixels / 4;

    bool can_read = true;

    can_read= can_read && yuv_f.read(reinterpret_cast<char*>(y), Ypixels * sizeof(T));
    can_read= can_read && yuv_f.read(reinterpret_cast<char*>(u), UVpixels * sizeof(T));
    can_read= can_read && yuv_f.read(reinterpret_cast<char*>(v), UVpixels * sizeof(T));
    
    return can_read;

}

template<class  T>
static void write_yuv420_frame(
    std::ofstream& out,
    T* y, T* u, T* v,
    int width,
    int height
) {
    int Ypixels = width * height;
    int UVpixels = width / 4;


    out.write(reinterpret_cast<const char*>(y), Ypixels * sizeof(T));
    out.write(reinterpret_cast<const char*>(u), UVpixels * sizeof(T));
    out.write(reinterpret_cast<const char*>(v), UVpixels * sizeof(T));
}



static void write_yuv420_AVFrame(std::ofstream& out,AVFrame* frame ) {
    int width = frame->width;
    int height = frame->height;

    for (int i = 0; i < height; i++) out.write((char*)frame->data[0] + i * frame->linesize[0], width);             // Y
    for (int i = 0; i < height / 2; i++) out.write((char*)frame->data[1] + i * frame->linesize[1], width / 2);     // U
    for (int i = 0; i < height / 2; i++) out.write((char*)frame->data[2] + i * frame->linesize[2], width / 2);     // V
 
}

#endif // !HELPER_H
