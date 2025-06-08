// nvdec_hevc_decode.cpp
extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/hwcontext.h>
#include <libavutil/imgutils.h>
#include <libswscale/swscale.h>
#include <libavutil/opt.h>
}

#include <iostream>
#include "kernel.cuh"
#include "yuv_file_helper.h"
#include <chrono>
#include <string>
#include <iomanip>

#define APP_VERSION "1.0.0"


#define CPU_DECODER 0
#define GPU_DECODER 1

static string inputfilepath = "";
static int use_hardware_decoder = CPU_DECODER;
static int start_frame_index = 0;
static int nb_frames = 300; // number of frames to decode
static int nb_threads = 0; // When using CPU decoder, number of threads to use for decoding, 0 means use all available threads

//YUV inputs params
static int yuv_w = 3840;
static int yuv_h = 1920;
static int yuv_bd = 8;

static void parse_parameters(int argc, char* argv[]) {
    
    
    std::cout << "Version " << APP_VERSION << std::endl;

    if (argc > 1)
    {
        std::string arg = argv[1];
        if (arg == "-help")
        {
            cout << "-------| " << "-----------------------------------------------------------------------------" << endl;
            cout << "Params | " << "Description/Options" << endl;
            cout << "-------| " << "-----------------------------------------------------------------------------" << endl;
            cout << "-i     | " << "Filepath of the input YUV420 or HEVC/H.264 intra-only file" << endl;
            cout << "-w     | " << "Width of the YUV file [default=3840]" << endl;
            cout << "-h     | " << "Height of the YUV file [default=1920]" << endl;
            cout << "-dt    | " << "Decoder Type (0=CPU_DECODER, 1=GPU_DECODER, else YUV reader) [default=0]" << endl;
            cout << "-bd    | " << "Bit-depth of the YUV file (only 8 is supported ) [default=8]" << endl;
            cout << "-sf    | " << "Start frame index to decode (or read for YUV file) [default=0]" << endl;
            cout << "-nf    | " << "Number of frames to decode (or read for YUV file) [default=300]" << endl;
            cout << "-nt    | " << "Number of threads (used only for CPU_DECODER); set '0' to use all physical cores [default=0]" << endl;
            exit(0);
        }
    }
    else {
        cout << "Input paramter error. Type '-help' to see the paramter list." << endl;
        exit(0);
    }
    
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "-i" && i + 1 < argc) {
			inputfilepath = string(argv[++i]);
        }
        else if (arg == "-dt" && i + 1 < argc) {
            use_hardware_decoder = std::stoi(argv[++i]);
        }
        else if (arg == "-w" && i + 1 < argc) {
            yuv_w = std::stoi(argv[++i]);
        }
        else if (arg == "-h" && i + 1 < argc) {
            yuv_h = std::stoi(argv[++i]);
        }
        else if (arg == "-bd" && i + 1 < argc) {
            yuv_bd = std::stoi(argv[++i]);
        }
        else if (arg == "-sf" && i + 1 < argc) {
            start_frame_index = std::stoi(argv[++i]);
        }
        else if (arg == "-nf" && i + 1 < argc) {
            nb_frames = std::stoi(argv[++i]);
        }
        else if (arg == "-nt" && i + 1 < argc) {
            nb_threads = std::stoi(argv[++i]);
        }
    }
}


static enum AVPixelFormat get_hw_format(AVCodecContext* ctx, const enum AVPixelFormat* pix_fmts) {
    for (const AVPixelFormat* p = pix_fmts; *p != -1; p++) {
        if (*p == AV_PIX_FMT_CUDA || *p == AV_PIX_FMT_NV12) {
            return *p;
        }
    }
    std::cerr << "Failed to get HW surface format.\n";
    return AV_PIX_FMT_NONE;
}

int main(int argc, char* argv[]) {

	parse_parameters(argc, argv);
	int end_frame_index = start_frame_index + nb_frames;
    if (use_hardware_decoder == CPU_DECODER)
    {
        std::cout << "Use CPU decoder...\n";
    }
    else if (use_hardware_decoder == GPU_DECODER)
    {
        std::cout << "Use GPU decoder...\n";
    }
    else
    {
        std::cout << "Read YUV file...\n";
    }


        if (use_hardware_decoder != CPU_DECODER && use_hardware_decoder != GPU_DECODER)
        {

            //string yuvfilepath = "C:\\TEST_YUV\\ffmpeg-7.1.1-full_build\\bin\\DrivingInCountry_3840x1920_30fps_8bit_420_erp.yuv";
			string yuvfilepath = inputfilepath;
            
            int W = yuv_w;
            int H = yuv_h;
            int bd = yuv_bd;

            //C++ 17 or higher
            using DTYPE = unsigned char;
            if (yuv_bd == 10) {
                using DTYPE = unsigned short;
            }

            ifstream yuv_f = open_YUV420_file(yuvfilepath, W, H, yuv_bd, start_frame_index);

            DTYPE* Y_img = (DTYPE*)_aligned_malloc(W * H * sizeof(DTYPE), 32);
            DTYPE* U_img = (DTYPE*)_aligned_malloc(W / 2 * H / 2 * sizeof(DTYPE), 32);
            DTYPE* V_img = (DTYPE*)_aligned_malloc(W / 2 * H / 2 * sizeof(DTYPE), 32);

            //DTYPE* Y_img = (DTYPE*)malloc(W * H * sizeof(DTYPE));
            //DTYPE* U_img = (DTYPE*)malloc(W / 2 * H / 2 * sizeof(DTYPE));
            //DTYPE* V_img = (DTYPE*)malloc(W / 2 * H / 2 * sizeof(DTYPE));

            int current_frame_inedx = start_frame_index;
            auto st = std::chrono::high_resolution_clock::now();
           
            while (current_frame_inedx<end_frame_index &&  read_YUV420_frame(yuv_f, Y_img, U_img, V_img, W, H)) {

                //write_yuv420_frame(out_yuv_f, Y_img, U_img, V_img, W, H);
                //auto et = std::chrono::high_resolution_clock::now();
                //auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(et - st).count() / 1000.0;
                //std::cout << "decoded time=" << current_frame_inedx << ", " << duration << " sec" << std::endl;

                current_frame_inedx ++;

            }
            auto et = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(et - st).count() / 1000.0;
            std::cout << "Number of decoded frames=" << current_frame_inedx - start_frame_index << std::endl;
            std::cout << "Average decoding time = " << std::setprecision(4) << (current_frame_inedx - start_frame_index) / duration << " fps" << std::endl;

            return 0;
        }

    //string hevcfilepath = "C:\\TEST_YUV\\ffmpeg-7.1.1-full_build\\bin\\DrivingInCountry_3840x1920_30fps_8bit_420_erp_crf18.mp4";
    string hevcfilepath = inputfilepath;

	//GPU or CPU decoder.....
    AVFormatContext* fmt_ctx = nullptr;
    int ret = avformat_open_input(&fmt_ctx, hevcfilepath.c_str(), nullptr, nullptr);
    if (ret < 0) {
        std::cout << "Cannot open the video file.\n";
        return -1;
    }
    ret = avformat_find_stream_info(fmt_ctx, nullptr);
    if (ret < 0) {
        std::cout << "Cannot find the stream.\n";
		return -1;
    }

    int video_stream_index = av_find_best_stream(fmt_ctx, AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
    //OR
    /*int video_stream_index = -1;
    for (unsigned int i = 0; i < fmt_ctx->nb_streams; i++) {
        if (fmt_ctx->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
            video_stream_index = i;
            break;
        }
    }*/

    if (video_stream_index < 0) {
        std::cerr << "Could not find video stream in file.\n";
        return -1;
    }

	//Start timing the decoding process
    auto st = std::chrono::high_resolution_clock::now();

	AVBufferRef* hw_device_ctx = nullptr;  //This is used if use_hardware_decoder is true
    const AVCodec* decoder = nullptr;

    AVStream* stream = fmt_ctx->streams[video_stream_index];
    decoder = avcodec_find_decoder(stream->codecpar->codec_id);
    //OR
    //AVCodecParameters* codecpar = fmt_ctx->streams[video_stream_index]->codecpar;
    //decoder = avcodec_find_decoder(codecpar->codec_id);
   
    if (!decoder) {
        std::cerr << "Decoder not found.\n";
        return -1;
    }

    AVCodecContext* codec_ctx = avcodec_alloc_context3(decoder);
    avcodec_parameters_to_context(codec_ctx, stream->codecpar);

    //Use GPU
    if (use_hardware_decoder==GPU_DECODER) {
        if (av_hwdevice_ctx_create(&hw_device_ctx, AV_HWDEVICE_TYPE_CUDA, nullptr, nullptr, 0) < 0) {
            std::cerr << "Failed to create CUDA device.\n";
            return -1;
        }
        codec_ctx->hw_device_ctx = av_buffer_ref(hw_device_ctx);
        codec_ctx->get_format = get_hw_format;
    }
    else
    {
        codec_ctx->thread_count = nb_threads;             // or 0 to let FFmpeg choose
        codec_ctx->thread_type = FF_THREAD_FRAME; //best for Intra only [other option: FF_THREAD_SLICE]
    }

    if (avcodec_open2(codec_ctx, decoder, nullptr) < 0) {
        std::cerr << "Failed to open codec.\n";
        return -1;
    }

    AVPacket* pkt = av_packet_alloc();
    AVFrame* frame = av_frame_alloc();
    AVFrame* sw_frame = av_frame_alloc();

    //Seek an optional frame
    int64_t target_pts = av_rescale_q(start_frame_index, { 1, stream->r_frame_rate.num }, stream->time_base);
    av_seek_frame(fmt_ctx, video_stream_index, target_pts, AVSEEK_FLAG_BACKWARD);

    // Create output frame (YUV420P)
    AVFrame* yuv420p_frame = av_frame_alloc();
    yuv420p_frame->format = AV_PIX_FMT_YUV420P;
    yuv420p_frame->width = codec_ctx->width;
    yuv420p_frame->height = codec_ctx->height;
    av_frame_get_buffer(yuv420p_frame, 32);  // aligned buffer

    int current_frame_inedx = start_frame_index;
    while (av_read_frame(fmt_ctx, pkt) >= 0 && current_frame_inedx < end_frame_index) {
        if (pkt->stream_index == video_stream_index) {
            if (avcodec_send_packet(codec_ctx, pkt) == 0) {
                while (avcodec_receive_frame(codec_ctx, frame) == 0 && current_frame_inedx < end_frame_index) {
                    // If the frame is on GPU memory, move it to CPU
                    if (frame->format == AV_PIX_FMT_CUDA) {
                        
                        //This is just for a test of modify the Y-ch using CUDA kernel
                        //uint8_t* d_y_plane = frame->data[0];  // already on GPU memory
                        //call_cuda_kernel(d_y_plane, frame->width, frame->height, frame->linesize[0]);

                        //********************* NOTE ******************************
                        //Because the pixel format is AV_PIX_FMT_CUDA frame->linesize[1] frame->linesize[1] includes ....
                        // ... both U and V data and frame->linesize[2] is zero.
                        //AV_PIX_FMT_CUDA is actually the same as AV_PIX_FMT_NV12 format in which U-V data for each pixel...
                        // ... are stored in one single array next to each other: arrayUV=[U0,V0,U1,V1,...]

                        //Move data from GPU to cpu memory
                        //The pixel format sw_frame is still AV_PIX_FMT_NV12=23
                        if (av_hwframe_transfer_data(sw_frame, frame, 0) < 0) {
                            std::cerr << "Error transferring frame from GPU.\n";
                            continue;
                        }

                        //sw_frame->format is AV_PIX_FMT_NV12=23, it must converted to AV_PIX_FMT_YUV420P if we need standard order of yuv file format
                        //save_yuv420_frame(sw_frame, sw_frame->width, sw_frame->height, "d:\\frame_12.yuv");

                        //------------ Convert AV_PIX_FMT_NV12 to AV_PIX_FMT_YUV420P ------------
                        //Step 1
                        // Create SwsContext
                        struct SwsContext* sws_ctx = sws_getContext(
                            sw_frame->width,
                            sw_frame->height,
                            (AVPixelFormat)sw_frame->format,      // input is NV12
                            sw_frame->width,
                            sw_frame->height,
                            AV_PIX_FMT_YUV420P,                   // output format
                            SWS_BILINEAR, nullptr, nullptr, nullptr);

                        //Step 2
                        // Convert
                        sws_scale(
                            sws_ctx,
                            sw_frame->data,
                            sw_frame->linesize,
                            0,
                            sw_frame->height,
                            yuv420p_frame->data,
                            yuv420p_frame->linesize);

						
                        //std::cout << "Decoded frame idx (GPU->CPU)="<< current_frame_inedx <<"\n";
                        //std::string outfilepath = "d:\\gpu_frames\\frame_GPU_" + std::to_string(current_frame_inedx) + ".yuv";
                        //save_yuv420_frame(yuv420p_frame, yuv420p_frame->width, yuv420p_frame->height, outfilepath);
                        sws_freeContext(sws_ctx);

                        //return 0;

                    }
                    else {
						//Cool! even with using multi-threaded CPU decoder, the frames are decoded and buffered in the time oder
                        //std::cout << "Decoded frame idx (CPU)=" << frame->pts << "\n"; 
                        
                        //std::string outfilepath = "d:\\frame_GPU_" + std::to_string(current_frame_inedx) + ".yuv";
                        //save_yuv420_frame(frame, frame->width, frame->height,outfilepath);
                        
                        //return 0;
                        
                    }
                    //auto et = std::chrono::high_resolution_clock::now();
                    //auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(et - st).count() / 1000.0;
                    //std::cout << "decoded time=" << current_frame_inedx <<", " << duration <<" sec" << std::endl;
                    

                    current_frame_inedx++;
                    
                }
            }
        }
        av_packet_unref(pkt);
    }
    auto et = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(et - st).count() / 1000.0;
    std::cout << "Number of decoded frames=" <<current_frame_inedx - start_frame_index << std::endl;
    std::cout << "Average decoding time = "<< std::setprecision(4)  << (current_frame_inedx - start_frame_index)/duration << " fps" << std::endl;

    av_frame_free(&frame);
    av_frame_free(&sw_frame);
    av_packet_free(&pkt);
    avcodec_free_context(&codec_ctx);
    avformat_close_input(&fmt_ctx);
    av_buffer_unref(&hw_device_ctx);
    av_frame_free(&yuv420p_frame);

    return 0;
}