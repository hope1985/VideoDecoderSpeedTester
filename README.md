# VideoDecoderSpeedTester
A project to compare video decoding speeds between the CPU and GPU (CUDA) using FFmpeg APIs.

# Requirements
- C++17 compatible compiler
- CUDA 

# Command-Line Parameters
| Params | Description/Options                                                                 |
|--------|--------------------------------------------------------------------------------------|
| `-i`   | Filepath of the input YUV420 or HEVC/H.264 intra-only file                         |
| `-w`   | Width of the YUV file (default: `3840`)                                             |
| `-h`   | Height of the YUV file (default: `1920`)                                            |
| `-dt`  | Decoder Type: `0` = CPU_DECODER, `1` = GPU_DECODER, others = YUV reader (default: `0`) |
| `-bd`  | Bit-depth of the YUV file (only `8` is supported) (default: `8`)                   |
| `-sf`  | Start frame index to decode or read (default: `0`)                                 |
| `-nf`  | Number of frames to decode or read (default: `300`)                                |
| `-nt`  | Number of threads (only used with CPU_DECODER); use `0` to utilize all physical cores (default: `0`) |

# Usage Example 
## CPU decoder
  <pre><code class="language-bash"> -dt 0  -nt 10 -sf 20 -nf 200 -i input.mp4 </code></pre>
## GPU decoder
  <pre><code class="language-bash"> -dt 1  -sf 20 -nf 200 -i input.mp4 </code></pre> 
## YUV reader
  <pre><code class="language-bash"> -dt 1 -sf 20 -nf 200 -w 3840 -h 1920 -bd 8 -i input.yuv </code></pre>
