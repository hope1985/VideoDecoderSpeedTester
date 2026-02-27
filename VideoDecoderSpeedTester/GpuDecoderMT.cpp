#include "GpuDecoderMT.h"
#include <mutex>

GpuDecoderMT::GpuDecoderMT() {
    av_log_set_level(AV_LOG_ERROR);

}

GpuDecoderMT::~GpuDecoderMT() {
    free_cuda_memory(&devY);
    free_cuda_memory(&devU);
    free_cuda_memory(&devV);
    close();
}

void GpuDecoderMT::producer() {


    while (!stop_flag.load()) {
        int idx = write_idx.load() % BUFFER_SIZE;

        // Wait until buffer is empty (i.e., status == 0)
        while (status[idx].load(std::memory_order_acquire) != 0)
        {
            if (stop_flag.load()) return;
            std::this_thread::sleep_for(std::chrono::microseconds(1));
        }// or sleep, or use condition_variable

        status[idx].store(1, std::memory_order_release); // mark as writing
        // Write data
        if (decode_next_frame(idx))
        {
            status[idx].store(2);  // mark as ready (full)
            write_idx.store((idx + 1) % BUFFER_SIZE, std::memory_order_release);


        }

    }
}




bool GpuDecoderMT::open(int decoderID, const std::string& filepath, int start_frame_index, int num_frames) {
    start_frame_index_ = start_frame_index;
    end_frame_index_ = num_frames > 0 ? (start_frame_index_ + num_frames) : INT32_MAX;
    id = decoderID;

    if (avformat_open_input(&fmt_ctx_, filepath.c_str(), nullptr, nullptr) < 0) {
        std::cerr << "Cannot open the video file.\n";
        return false;
    }

    if (avformat_find_stream_info(fmt_ctx_, nullptr) < 0) {
        std::cerr << "Cannot find the stream info.\n";
        return false;
    }

    video_stream_index_ = av_find_best_stream(fmt_ctx_, AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
    if (video_stream_index_ < 0) {
        std::cerr << "Cannot find video stream.\n";
        return false;
    }

    video_stream_ = fmt_ctx_->streams[video_stream_index_];
    decoder_ = avcodec_find_decoder(video_stream_->codecpar->codec_id);
    if (!decoder_) {
        std::cerr << "Decoder not found.\n";
        return false;
    }

    codec_ctx_ = avcodec_alloc_context3(decoder_);
    avcodec_parameters_to_context(codec_ctx_, video_stream_->codecpar);


    // the third parameter is the GPU id e.g., "0": for the first GPU, "1" for the second GPU, etc.
    if (av_hwdevice_ctx_create(&hw_device_ctx_, AV_HWDEVICE_TYPE_CUDA, nullptr, nullptr, 0) < 0) {
        std::cerr << "Failed to create CUDA device.\n";
        return false;
    }

    codec_ctx_->hw_device_ctx = av_buffer_ref(hw_device_ctx_);
    codec_ctx_->get_format = get_hw_format;


    // Allocate a GPU frame pool with a usable pixel format (e.g., NV12 or P010LE)
    /*hw_frames_ctx = av_hwframe_ctx_alloc(hw_device_ctx_);
    AVHWFramesContext* frames_ctx = (AVHWFramesContext*)hw_frames_ctx->data;

    frames_ctx->format = AV_PIX_FMT_CUDA;             // decoder outputs
    frames_ctx->sw_format = AV_PIX_FMT_P010LE;        // mapped output format (still GPU)
    frames_ctx->width = codec_ctx_->width;
    frames_ctx->height = codec_ctx_->height;
    frames_ctx->initial_pool_size = 10;

    av_hwframe_ctx_init(hw_frames_ctx);*/

    //Open Decoder
    if (avcodec_open2(codec_ctx_, decoder_, nullptr) < 0) {
        std::cerr << "Failed to open codec.\n";
        return false;
    }

    width = codec_ctx_->width;
    height = codec_ctx_->height;
    pkt_ = av_packet_alloc();

    frame_ = av_frame_alloc();
    frame_1 = av_frame_alloc();
    sw_frame_ = av_frame_alloc();
    mapped_frame = av_frame_alloc();
    mapped_frame1 = av_frame_alloc();
    /*auto s = std::string("d:\\outputall") + std::string(".yuv");
    fp = fopen(s.c_str(), "wb");
    if (!fp) {
        perror("Cannot open file for writing");

    }*/

    //For test
   // int64_t seek_pts = av_rescale_q(start_frame_index_, { 1, video_stream_->r_frame_rate.num }, video_stream_->time_base);
   // av_seek_frame(fmt_ctx_, video_stream_index_, seek_pts, AVSEEK_FLAG_BACKWARD);

    current_frame_index_ = start_frame_index_;


    frames[0] = av_frame_alloc();
    frames[1] = av_frame_alloc();

    //init cuda memmoy
    init_cuda_memory(&devY, codec_ctx_->width, codec_ctx_->height, sizeof(DTYPE));
    init_cuda_memory(&devU, codec_ctx_->width / 2, codec_ctx_->height / 2, sizeof(DTYPE));
    init_cuda_memory(&devV, codec_ctx_->width / 2, codec_ctx_->height / 2, sizeof(DTYPE));

    init_cuda_memory(&devY1, codec_ctx_->width, codec_ctx_->height, sizeof(DTYPE));
    init_cuda_memory(&devU1, codec_ctx_->width / 2, codec_ctx_->height / 2, sizeof(DTYPE));
    init_cuda_memory(&devV1, codec_ctx_->width / 2, codec_ctx_->height / 2, sizeof(DTYPE));

    /*auto s = std::string("c:\\a\\outputv1") + std::to_string(id) + std::string(".yuv");
    fp1 = fopen(s.c_str(), "wb");
    if (!fp1) {
        perror("Cannot open file for writing");

    }

    fp2 = fp1;*/
    //if(id==0)
        prod = new std::thread(&GpuDecoderMT::producer, this);

    return true;
}

//output_frame is not used, but it is used it should be deleted after use
bool GpuDecoderMT::decode_next_frame(int buff_idx) {

    if (current_frame_index_ > end_frame_index_)
    {
        return false;
        std::cerr << "cannot read frame with index higher 'end_frame_index'.\n";
    }


    //Necessaery to get the data from the correct stream if there are multiple audio/video streams 
    //but if there is only one video stream it will be always one video stream and this can be removed!
    while (av_read_frame(fmt_ctx_, pkt_) >= 0) {
        if (pkt_->stream_index != video_stream_index_) {
            av_packet_unref(pkt_);
            continue;
        }
        //No need to allocate memory for each decoded frame, it is already allocated in the open() method
        //frame_ = av_frame_alloc();
        //auto frame2 = av_frame_alloc();
        if (avcodec_send_packet(codec_ctx_, pkt_) == 0) {
            while (avcodec_receive_frame(codec_ctx_, frames[buff_idx]) == 0 && current_frame_index_ < end_frame_index_) {
                if (frames[buff_idx]->format == AV_PIX_FMT_CUDA) {


                    // Allocate GPU memory for mapped_frame using hardware frame context
                    /*if (av_hwframe_get_buffer(hw_frames_ctx, mapped_frame, 0) < 0) {
                        fprintf(stderr, "Failed to allocate mapped GPU frame\n");
                        return -1;
                    }


                    // Transfer the data layout into mapped_frame
                    if (av_hwframe_transfer_data(mapped_frame, frame_, 0) < 0) {
                        fprintf(stderr, "Failed to map hardware frame to GPU-accessible format\n");
                        return -1;
                    }*/



                    if (buff_idx == 0)
                    {
                        //This moves the frame to CPU memory. However it is not used, it is necessary because sometime the gpu buffer is not ready to read and this solve this problem
                        av_hwframe_transfer_data(mapped_frame, frames[buff_idx], 0);

                        convert_nv12_10bit_yuv420_10bit_le(frames[buff_idx], (uint8_t*)devY, (uint8_t*)devU, (uint8_t*)devV);
                    }
                    if (buff_idx == 1)
                    {
                        //This moves the frame to CPU memory. However it is not used, it is necessary because sometime the gpu buffer is not ready to read and this solve this problem
                        av_hwframe_transfer_data(mapped_frame1, frames[buff_idx], 0);
                        convert_nv12_10bit_yuv420_10bit_le(frames[buff_idx], (uint8_t*)devY1, (uint8_t*)devU1, (uint8_t*)devV1);
                    }


                    // ======== CONVERT NV12 TO YUV420 ON GPU ========
                    if (buff_idx == 2)
                    {
                        //This moves the frame to CPU memory. However it is not used, it is necessary because sometime the gpu buffer is not ready to read and this solve this problem
                        av_hwframe_transfer_data(mapped_frame, frames[buff_idx], 0);

                        convert_nv12_10bit_yuv420_10bit_le(frames[buff_idx], (uint8_t*)devY, (uint8_t*)devU, (uint8_t*)devV);

                        //if (id == 0)
                        {
                            // ======== CONVERT FRAME FORMAT TO AV_PIX_FMT_YUV420P ========
                        // Create SwsContext
                          /* struct SwsContext* sws_ctx = sws_getContext(
                                mapped_frame->width,
                                mapped_frame->height,
                                (AVPixelFormat)mapped_frame->format,      // input is NV12
                                mapped_frame->width,
                                mapped_frame->height,
                                AV_PIX_FMT_YUV420P10LE,                   // output format
                                SWS_BILINEAR, nullptr, nullptr, nullptr);

                            //Do not need do it each time
                            auto output_frame = av_frame_alloc();
                            output_frame->format = AV_PIX_FMT_YUV420P10LE;
                            output_frame->width = mapped_frame->width;
                            output_frame->height = mapped_frame->height;

                            av_frame_get_buffer(output_frame, 32);

                            // Convert
                            sws_scale(
                                sws_ctx,
                                mapped_frame->data,
                                mapped_frame->linesize,
                                0,
                                mapped_frame->height,
                                output_frame->data,
                                output_frame->linesize);

                            sws_freeContext(sws_ctx_);

                            DTYPE* hostY =(DTYPE*)  output_frame->data[0]; //new DTYPE[y_size];
                            DTYPE* hostU = (DTYPE*) output_frame->data[1]; // new DTYPE[uv_size];
                            DTYPE* hostV = (DTYPE*) output_frame->data[2];  //new DTYPE[uv_size];
                            */


                            int y_size = frames[buff_idx]->width * frames[buff_idx]->height;
                            int uv_size = (frames[buff_idx]->width / 2) * (frames[buff_idx]->height / 2);

                            DTYPE* hostY = new DTYPE[y_size];
                            DTYPE* hostU = new DTYPE[uv_size];
                            DTYPE* hostV = new DTYPE[uv_size];
                            
                            cudaMemcpy(hostY, devY, y_size * sizeof(DTYPE), cudaMemcpyDeviceToHost);
                            cudaMemcpy(hostU, devU, uv_size * sizeof(DTYPE), cudaMemcpyDeviceToHost);
                            cudaMemcpy(hostV, devV, uv_size * sizeof(DTYPE), cudaMemcpyDeviceToHost);

     

                            fwrite(hostY, sizeof(DTYPE), y_size, fp1);
                            fwrite(hostU, sizeof(DTYPE), uv_size, fp1);
                            fwrite(hostV, sizeof(DTYPE), uv_size, fp1);

                            delete hostY;
                            delete hostU;
                            delete hostV;

                           // fclose(fp1);
                        }
                    }
                    if (buff_idx == 3)
                    {

                        //This moves the frame to CPU memory. However it is not used, it is necessary because sometime the gpu buffer is not ready to read and this solve this problem
                        av_hwframe_transfer_data(mapped_frame1, frames[buff_idx], 0);


                        convert_nv12_10bit_yuv420_10bit_le(frames[buff_idx], (uint8_t*)devY1, (uint8_t*)devU1, (uint8_t*)devV1);
                        //if (id == 0)
                        {

                            // ======== CONVERT FRAME FORMAT TO AV_PIX_FMT_YUV420P ========
                        // Create SwsContext
                           /* struct SwsContext* sws_ctx = sws_getContext(
                                mapped_frame1->width,
                                mapped_frame1->height,
                                (AVPixelFormat)mapped_frame1->format,      // input is NV12
                                mapped_frame1->width,
                                mapped_frame1->height,
                                AV_PIX_FMT_YUV420P10LE,                   // output format
                                SWS_BILINEAR, nullptr, nullptr, nullptr);

                            //Do not need do it each time
                            auto output_frame = av_frame_alloc();
                            output_frame->format = AV_PIX_FMT_YUV420P10LE;
                            output_frame->width = mapped_frame1->width;
                            output_frame->height = mapped_frame1->height;

                            av_frame_get_buffer(output_frame, 32);

                            // Convert
                            sws_scale(
                                sws_ctx,
                                mapped_frame1->data,
                                mapped_frame1->linesize,
                                0,
                                mapped_frame1->height,
                                output_frame->data,
                                output_frame->linesize);

                            sws_freeContext(sws_ctx_);

                      

                            DTYPE* hostY = (DTYPE*)output_frame->data[0]; //new DTYPE[y_size];
                            DTYPE* hostU = (DTYPE*)output_frame->data[1]; // new DTYPE[uv_size];
                            DTYPE* hostV = (DTYPE*)output_frame->data[2];  //new DTYPE[uv_size];
                            */


                            int y_size = frames[buff_idx]->width * frames[buff_idx]->height;
                            int uv_size = (frames[buff_idx]->width / 2) * (frames[buff_idx]->height / 2);

                            DTYPE* hostY = new DTYPE[y_size];
                            DTYPE* hostU = new DTYPE[uv_size];
                            DTYPE* hostV = new DTYPE[uv_size];
                            cudaMemcpy(hostY, devY1, y_size * sizeof(DTYPE), cudaMemcpyDeviceToHost);
                            cudaMemcpy(hostU, devU1, uv_size * sizeof(DTYPE), cudaMemcpyDeviceToHost);
                            cudaMemcpy(hostV, devV1, uv_size * sizeof(DTYPE), cudaMemcpyDeviceToHost);



                            fwrite(hostY, sizeof(DTYPE), y_size, fp2);
                            fwrite(hostU, sizeof(DTYPE), uv_size, fp2);
                            fwrite(hostV, sizeof(DTYPE), uv_size, fp2);


                            delete hostY;
                            delete hostU;
                            delete hostV;
                            //fclose(fp2);
                        }
                    }

                    //auto now = std::chrono::system_clock::now();
                    //auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch());

                    //std::cout << "dec" <<id<< " - Milliseconds since epoch: " << ms.count() << " ms" << std::endl;
                    //frames.push_back(mapped_frame);

                    //=========== For test



                    // ======== COPY FRAME FROM GPU TO CPU ========
                    /*if (av_hwframe_transfer_data(sw_frame_, frame_, 0) < 0) {
                        std::cerr << "Error transferring frame from GPU.\n";
                        av_packet_unref(pkt_);
                        return false;
                    }*/

                    //=========== For test
                      /*FILE* fp2 = fopen("d:\\output.yuv", "wb");
                      int yls = (sw_frame_->linesize[0] / 2);
                      int uls = (sw_frame_->linesize[0] / 2);
                      for (int i = 0; i < sw_frame_->height; i++)
                      {
                          fwrite((DTYPE*)sw_frame_->data[0] + i * yls, sizeof(DTYPE), sw_frame_->width, fp2);
                          fwrite((DTYPE*)sw_frame_->data[1] + i * uls, sizeof(DTYPE), sw_frame_->width, fp2);
                          fwrite((DTYPE*)sw_frame_->data[1] + uls+i * uls, sizeof(DTYPE), sw_frame_->width, fp2);
                      }
                      fclose(fp2);*/


                      // ======== CONVERT FRAME FORMAT TO AV_PIX_FMT_YUV420P ========
                      /*sws_ctx_ = sws_getContext(
                          sw_frame_->width,
                          sw_frame_->height,
                          static_cast<AVPixelFormat>(sw_frame_->format),
                          sw_frame_->width,
                          sw_frame_->height,
                          AV_PIX_FMT_YUV420P10LE,
                          SWS_BILINEAR, nullptr, nullptr, nullptr);

                      //Do not need do it each time
                      output_frame->format = AV_PIX_FMT_YUV420P10LE;
                      output_frame->width = sw_frame_->width;
                      output_frame->height = sw_frame_->height;
                      av_frame_get_buffer(output_yuv4210ple_frame, 32);

                      sws_scale(sws_ctx_,
                          sw_frame_->data,
                          sw_frame_->linesize,
                          0,
                          sw_frame_->height,
                          output_frame->data,
                          output_frame->linesize);

                      sws_freeContext(sws_ctx_);

                      FILE* fp2 = fopen("d:\\output2.yuv", "wb");
                      if (!fp2) {
                          perror("Cannot open file for writing");

                      }

                      // Write in planar order: Y -> U -> V
                      fwrite(output_yuv4210ple_frame->data[0], sizeof(uint16_t), y_size, fp2);
                      fwrite(output_yuv4210ple_frame->data[1], sizeof(uint16_t), uv_size, fp2);
                      fwrite(output_yuv4210ple_frame->data[2], sizeof(uint16_t), uv_size, fp2);

                      fclose(fp2);*/

                      //No need to deallocate memory for each decoded frame, it is deallocated in the close() method
                      //av_frame_free(&frame_);

                    //av_packet_unref(pkt_);
                    current_frame_index_++;
                    return true;
                }
            }
        }
        av_packet_unref(pkt_);
    }

    return false; // End of file or error
}





//output_frame is not used, but it is used it should be deleted after use
bool GpuDecoderMT::decode_next_frame(/*AVFrame* output_frame*/) {

    // if (frames.size() > fidx )
    // {
         //frames[fidx] =...

         //return;
    // }
    // 


    if (current_frame_index_ > end_frame_index_)
    {
        return false;
        std::cerr << "cannot read frame with index higher 'end_frame_index'.\n";
    }

    //Necessaery to get the data from the correct stream if there are multiple audio/video streams 
    //but if there is only one video stream it will be always one video stream and this can be removed!
    while (av_read_frame(fmt_ctx_, pkt_) >= 0) {
        if (pkt_->stream_index != video_stream_index_) {
            av_packet_unref(pkt_);
            continue;
        }
        //No need to allocate memory for each decoded frame, it is already allocated in the open() method
        //frame_ = av_frame_alloc();
        if (avcodec_send_packet(codec_ctx_, pkt_) == 0) {
            int idx_buf = current_frame_index_ % 2;
            while (avcodec_receive_frame(codec_ctx_, frame_) == 0 && current_frame_index_ < end_frame_index_) {
                if (frame_->format == AV_PIX_FMT_CUDA) {


                    // Allocate GPU memory for mapped_frame using hardware frame context
                    /*if (av_hwframe_get_buffer(hw_frames_ctx, mapped_frame, 0) < 0) {
                        fprintf(stderr, "Failed to allocate mapped GPU frame\n");
                        return -1;
                    }


                    // Transfer the data layout into mapped_frame
                    if (av_hwframe_transfer_data(mapped_frame, frame_, 0) < 0) {
                        fprintf(stderr, "Failed to map hardware frame to GPU-accessible format\n");
                        return -1;
                    }*/

                    //This moves the frame to CPU memory. However it is not used, it is necessary because sometime the gpu buffer is not ready to read and this solve this problem
                    av_hwframe_transfer_data(mapped_frame, frame_, 0);

                    // ======== CONVERT NV12 TO YUV420 ON GPU ========
                    convert_nv12_10bit_yuv420_10bit_le(frame_, (uint8_t*)devY, (uint8_t*)devU, (uint8_t*)devV);
                    //frames.push_back(mapped_frame);

                    //=========== For test
                    /*int y_size = frame_->width * frame_->height;
                    int uv_size = (frame_->width / 2) * (frame_->height / 2);


                    DTYPE* hostY = new DTYPE[y_size];
                    DTYPE* hostU = new DTYPE[uv_size];
                    DTYPE* hostV = new DTYPE[uv_size];

                    cudaMemcpy(hostY, devY, y_size * sizeof(DTYPE), cudaMemcpyDeviceToHost);
                    cudaMemcpy(hostU, devU, uv_size * sizeof(DTYPE), cudaMemcpyDeviceToHost);
                    cudaMemcpy(hostV, devV, uv_size * sizeof(DTYPE), cudaMemcpyDeviceToHost);

                    auto s = std::string("d:\\output") + std::string(".yuv");
                    FILE* fp = fopen(s.c_str(), "wb");
                    if (!fp) {
                        perror("Cannot open file for writing");

                    }

                    //Write in planar order: Y -> U -> V
                    //fwrite(hostY, sizeof(DTYPE), y_size, fp);
                    //fwrite(hostU, sizeof(DTYPE), uv_size, fp);
                    //fwrite(hostV, sizeof(DTYPE), uv_size, fp);
                    //fclose(fp);  */

                    // ======== COPY FRAME FROM GPU TO CPU ========
                    /*if (av_hwframe_transfer_data(sw_frame_, frame_, 0) < 0) {
                        std::cerr << "Error transferring frame from GPU.\n";
                        av_packet_unref(pkt_);
                        return false;
                    }*/

                    //=========== For test
                      /*FILE* fp2 = fopen("d:\\output.yuv", "wb");
                      int yls = (sw_frame_->linesize[0] / 2);
                      int uls = (sw_frame_->linesize[0] / 2);
                      for (int i = 0; i < sw_frame_->height; i++)
                      {
                          fwrite((DTYPE*)sw_frame_->data[0] + i * yls, sizeof(DTYPE), sw_frame_->width, fp2);
                          fwrite((DTYPE*)sw_frame_->data[1] + i * uls, sizeof(DTYPE), sw_frame_->width, fp2);
                          fwrite((DTYPE*)sw_frame_->data[1] + uls+i * uls, sizeof(DTYPE), sw_frame_->width, fp2);
                      }
                      fclose(fp2);*/


                      // ======== CONVERT FRAME FORMAT TO AV_PIX_FMT_YUV420P ========
                      /*sws_ctx_ = sws_getContext(
                          sw_frame_->width,
                          sw_frame_->height,
                          static_cast<AVPixelFormat>(sw_frame_->format),
                          sw_frame_->width,
                          sw_frame_->height,
                          AV_PIX_FMT_YUV420P10LE,
                          SWS_BILINEAR, nullptr, nullptr, nullptr);

                      //Do not need do it each time
                      output_frame->format = AV_PIX_FMT_YUV420P10LE;
                      output_frame->width = sw_frame_->width;
                      output_frame->height = sw_frame_->height;
                      av_frame_get_buffer(output_yuv4210ple_frame, 32);

                      sws_scale(sws_ctx_,
                          sw_frame_->data,
                          sw_frame_->linesize,
                          0,
                          sw_frame_->height,
                          output_frame->data,
                          output_frame->linesize);

                      sws_freeContext(sws_ctx_);

                      FILE* fp2 = fopen("d:\\output2.yuv", "wb");
                      if (!fp2) {
                          perror("Cannot open file for writing");

                      }

                      // Write in planar order: Y -> U -> V
                      fwrite(output_yuv4210ple_frame->data[0], sizeof(uint16_t), y_size, fp2);
                      fwrite(output_yuv4210ple_frame->data[1], sizeof(uint16_t), uv_size, fp2);
                      fwrite(output_yuv4210ple_frame->data[2], sizeof(uint16_t), uv_size, fp2);

                      fclose(fp2);*/

                      //No need to deallocate memory for each decoded frame, it is deallocated in the close() method
                      //av_frame_free(&frame_);

                    //av_packet_unref(pkt_);
                    current_frame_index_++;
                    return true;
                }
            }
        }
        av_packet_unref(pkt_);
    }

    return false; // End of file or error
}




void GpuDecoderMT::close() {
    if (fmt_ctx_) avformat_close_input(&fmt_ctx_);
    if (codec_ctx_) avcodec_free_context(&codec_ctx_);
    if (pkt_) av_packet_free(&pkt_);
    if (frame_) av_frame_free(&frame_);
    if (frame_1) av_frame_free(&frame_1);
    if (sw_frame_) av_frame_free(&sw_frame_);
    if (mapped_frame) av_frame_free(&mapped_frame);
    if (mapped_frame1) av_frame_free(&mapped_frame1);
    if (hw_device_ctx_) av_buffer_unref(&hw_device_ctx_);
    //fclose(fp);

    if (frames[0]) av_frame_free(&frames[0]);

    if (frames[1]) av_frame_free(&frames[1]);
    this->stop_flag.store(true);
    if (this->prod && this->prod->joinable()) {
        this->prod->join();
        delete this->prod;
        this->prod = nullptr;
    }
}

AVPixelFormat GpuDecoderMT::get_hw_format(AVCodecContext* ctx, const AVPixelFormat* pix_fmts) {
    while (*pix_fmts != AV_PIX_FMT_NONE) {
        if (*pix_fmts == AV_PIX_FMT_CUDA)
            return *pix_fmts;
        pix_fmts++;
    }
    std::cerr << "Failed to get HW surface format.\n";
    return AV_PIX_FMT_NONE;
}
