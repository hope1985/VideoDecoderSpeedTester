#include <omp.h>
#include <iostream>
#include <vector>
#include <string>

#include <chrono>
#include <iomanip>
#include <iostream>
#include <istream>
#include <fstream>
#include "GpuDecoderMT.h" // Your custom class
#include "yuv_file_helper.h" // Your custom helper functions


#include <iostream>
#include <fstream>
#include <vector>
#include <cstdint>
#include <cstring> // for memcpy
#include "YUVMemoryStream.h"

#ifdef USE_MULTI_THREAD_GPU_DECODER_CLASS
int main() {


    // List of input files (use different ones or the same for testing)

    /*std::vector<std::string> input_files = {
        "C:\\test_mp4\\content\\v1_texture_4096x2048_yuv420p10le-qp32-ldp-18.mp4",
        "C:\\test_mp4\\content\\v2_texture_4096x2048_yuv420p10le-qp32-ldp-18.mp4",
        "C:\\test_mp4\\content\\v3_texture_4096x2048_yuv420p10le-qp32-ldp-18.mp4",
        "C:\\test_mp4\\content\\v4_texture_4096x2048_yuv420p10le-qp32-ldp-18.mp4",
        //"C:\\test_mp4\\content\\v5_texture_4096x2048_yuv420p10le-qp32-ldp-18.mp4",
        //"C:\\test_mp4\\content\\v0_texture_4096x2048_yuv420p10le-qp32-ldp-18.mp4"
    };*/

    // List of input files (use different ones or the same for testing)
      std::vector<std::string> input_files = {
         "content\\v1_texture_4096x2048_yuv420p10le-qp32-ldp-18.mp4",
         "content\\v1_texture_4096x2048_yuv420p10le-qp32-ldp-18.mp4",
         "content\\v1_texture_4096x2048_yuv420p10le-qp32-ldp-18.mp4",
         "content\\v1_texture_4096x2048_yuv420p10le-qp32-ldp-18.mp4",
        // "content\\v1_texture_4096x2048_yuv420p10le-qp32-ldp-18.mp4",
        // "content\\v1_texture_4096x2048_yuv420p10le-qp32-ldp-18.mp4",
     };

     // List of input files (use different ones or the same for testing)
    /*std::vector<std::string> input_files = {
         "DrivingInCountry_3840x1920_30fps_8bit_420_erp-1.yuv",
         "DrivingInCountry_3840x1920_30fps_8bit_420_erp-2.yuv",
         "DrivingInCountry_3840x1920_30fps_8bit_420_erp-3.yuv",
         "DrivingInCountry_3840x1920_30fps_8bit_420_erp-4.yuv",
         "DrivingInCountry_3840x1920_30fps_8bit_420_erp-5.yuv",
         "DrivingInCountry_3840x1920_30fps_8bit_420_erp-6.yuv",
     };*/

    int num_decoders = input_files.size();

    int start_frame = 0;
    int nb_frame = 50;

    omp_set_num_threads(num_decoders);

    //std::cout << "num_threads:" << omp_get_num_threads() << std::endl;

    GpuDecoderMT* decoder = new GpuDecoderMT[num_decoders];
    AVFrame** frames = new AVFrame * [num_decoders];

    FILE** fps = new FILE * [num_decoders];
	YUVMemoryStream* memstream = new YUVMemoryStream[num_decoders];

    //std::ofstream** out_streams = new std::ofstream * [num_decoders];
    bool* can_read = new bool[num_decoders];
    for (int i = 0; i < num_decoders; ++i) {
        can_read[i] = false;

        if (!decoder[i].open(i, input_files[i], start_frame, nb_frame)) {
            std::cerr << "Thread " << i << ": Failed to open file.\n";
            return -1;
        }
        else
        {

            auto s = std::string("c:\\a\\output_v") + std::to_string(i) + std::string(".yuv");
            memstream[i].openFile(s);
            // fps[i] = fopen(s.c_str(), "wb");


            //std::string filename = "d:\\a\\output_" + std::to_string(i) + ".yuv";
            //out_streams[i] = new std::ofstream(filename, std::ios::binary);

            //frames[i] = av_frame_alloc();
            //we do not need to allocate memory for each read.
            //frames[i]->format = AV_PIX_FMT_YUV420P;
            //frames[i]->width = decoder[i].codec_ctx_->width;
            //frames[i]->height = decoder[i].codec_ctx_->height;
            //av_frame_get_buffer(frames[i], 32);
        }
    }

    auto st = std::chrono::high_resolution_clock::now();
    // This is parallel: each thread runs a separate decoder instance
    int frame_index = start_frame;

    for (; frame_index < start_frame + nb_frame; frame_index++)
    {

        for (int i = 0; i < num_decoders; ++i)
        {

            std::cout << "FRAME/DEC: " << std::setprecision(4) << frame_index << "/" << i << std::endl;
            // decoder[i]->decode_next_frame(frame_index%2);
            //std::this_thread::sleep_for(std::chrono::milliseconds(200));
            //=================================== CONSUMER ===================================
            int idx = decoder[i].read_idx.load(std::memory_order_acquire) % BUFFER_SIZE;

            // Wait until buffer is full (i.e., ready to read)
            while (decoder[i].status[idx].load(std::memory_order_acquire) != 2)
                std::this_thread::sleep_for(std::chrono::milliseconds(1));

            //========================================== PROCESS =====================================
            auto buff_idx = frame_index % 2;
            int y_size = decoder[i].width * decoder[i].height;
            int uv_size = (decoder[i].width / 2) * (decoder[i].height / 2);

            
            if (frame_index % 2 == 0)
            {

                DTYPE* hostY = new DTYPE[y_size];
                DTYPE* hostU = new DTYPE[uv_size];
                DTYPE* hostV = new DTYPE[uv_size];

                cudaMemcpy(hostY, decoder[i].devY, y_size * sizeof(DTYPE), cudaMemcpyDeviceToHost);
                cudaMemcpy(hostU, decoder[i].devU, uv_size * sizeof(DTYPE), cudaMemcpyDeviceToHost);
                cudaMemcpy(hostV, decoder[i].devV, uv_size * sizeof(DTYPE), cudaMemcpyDeviceToHost);

                //fwrite(hostY, sizeof(DTYPE), y_size, fps[i]);
                //fwrite(hostU, sizeof(DTYPE), uv_size, fps[i]);
                //fwrite(hostV, sizeof(DTYPE), uv_size, fps[i]);


                memstream[i].addFrame(reinterpret_cast<const uint8_t*>(hostY), y_size * sizeof(DTYPE));
                memstream[i].addFrame(reinterpret_cast<const uint8_t*>(hostU), uv_size * sizeof(DTYPE));
                memstream[i].addFrame(reinterpret_cast<const uint8_t*>(hostV), uv_size * sizeof(DTYPE));



                delete hostY;
                delete hostU;
                delete hostV;

            }
            if (frame_index % 2 == 1)
            {


 
                DTYPE* hostY = new DTYPE[y_size];
                DTYPE* hostU = new DTYPE[uv_size];
                DTYPE* hostV = new DTYPE[uv_size];

                cudaMemcpy(hostY, decoder[i].devY1, y_size * sizeof(DTYPE), cudaMemcpyDeviceToHost);
                cudaMemcpy(hostU, decoder[i].devU1, uv_size * sizeof(DTYPE), cudaMemcpyDeviceToHost);
                cudaMemcpy(hostV, decoder[i].devV1, uv_size * sizeof(DTYPE), cudaMemcpyDeviceToHost);

                memstream[i].addFrame(reinterpret_cast<const uint8_t*>(hostY), y_size * sizeof(DTYPE));
                memstream[i].addFrame(reinterpret_cast<const uint8_t*>(hostU), uv_size * sizeof(DTYPE));
                memstream[i].addFrame(reinterpret_cast<const uint8_t*>(hostV), uv_size * sizeof(DTYPE));


                //fwrite(hostY, sizeof(DTYPE), y_size, fps[i]);
                //fwrite(hostU, sizeof(DTYPE), uv_size, fps[i]);
                //fwrite(hostV, sizeof(DTYPE), uv_size, fps[i]);

                delete hostY;
                delete hostU;
                delete hostV;


            }
            if (memstream[i].totalSize() > (y_size + 2 * uv_size) * 20)
            {
				memstream[i].writeToFile();
            }
            
            //========================================== PROCES =====================================

            decoder[i].status[idx].store(0, std::memory_order_release); // mark as empty
            decoder[i].read_idx.store((idx + 1) % BUFFER_SIZE, std::memory_order_release);
        }
        //std::this_thread::sleep_for(std::chrono::milliseconds(50));
        //=================================== CONSUMER ===================================


        /*
        //#pragma omp parallel for
        for (int i = 0; i < num_decoders; ++i)
        {
            auto stdf = std::chrono::high_resolution_clock::now();
            can_read[i] = decoder[i].decode_next_frame(frames[i]);

            //while (decoder[i].decode_next_frame(frames[i]))
            {

                //write_yuv420_AVFrame( *out_streams[i], frames[i]);

                //do something with the decoded frame
                //#pragma omp critical
                  // std::cout << "Thread " << i << ": Decoded frame " << frame_index << "\n";
            }
            if (can_read[i])
            {
                auto etdf = std::chrono::high_resolution_clock::now();
                auto durationdf = std::chrono::duration_cast<std::chrono::milliseconds>(etdf - stdf).count() / 1000.0;
                std::cout << "frame decoding time for each video: " << std::setprecision(4) << durationdf << " seconds." << std::endl;
            }
            //std::cout << "frame decoding speed for each video: " << std::setprecision(4) << 1 / durationdf << " fps" << std::endl;
        }
        bool all_decoder_can_read = true;
        for (int i = 0; i < num_decoders; ++i)
        {
            all_decoder_can_read = all_decoder_can_read && can_read[i];
        }
        if (!all_decoder_can_read)
            break;*/
        // Write frames to output files
        //std::cout << "Frame index:" << frame_index << std::endl;
        //++frame_index;
        //auto etf = std::chrono::high_resolution_clock::now();
        //auto durationf = std::chrono::duration_cast<std::chrono::milliseconds>(etf - stf).count() / 1000.0;
        //std::cout << "frame decoding time for for all videos: " << std::setprecision(4) << durationf << " seconds." << std::endl;
        //std::cout << "frame decoding speed for all videos: " << std::setprecision(4) << 1 / durationf << " fps" << std::endl;
    }
    
    for (int i = 0; i < num_decoders; ++i) {
       
        memstream[i].writeToFile();

    }
    
    //frame_index--;
    //std::cout << "Number of decoded frames for each video: " << frame_index - start_frame << std::endl;

   /* for (int i = 0; i < num_decoders; ++i) {
        //av_frame_free(&frames[i]);
        //out_streams[i]->close();
        //delete[] out_streams[i];
    }
    delete[] frames;
    //delete[] out_streams;*/


    /*for (int i = 0; i < num_decoders; ++i)
    {
        fclose(fps[i]);
    }
    delete[] fps;*/
 

   // auto et = std::chrono::high_resolution_clock::now();
   // std::cout << "Number of videos:" << num_decoders << std::endl;
   // std::cout << "Number of decoded frames for each video: " << frame_index - start_frame << std::endl;
   // std::cout << "Number of total decoded frames for all videos: " << (num_decoders * (frame_index - start_frame)) << std::endl;
    //auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(et - st).count() / 1000.0;
   // std::cout << "Total decoding time for for all videos: " << std::setprecision(4) << duration << " seconds." << std::endl;
    //std::cout << "Average decoding speed for all videos: " << std::setprecision(4) << (num_decoders * (frame_index - start_frame)) / duration << " fps" << std::endl;


    return 0;
}
#endif