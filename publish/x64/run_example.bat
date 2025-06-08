VideoDecoderSpeedTester -help

VideoDecoderSpeedTester -dt 0  -nt 10 -sf 20 -nf 200 -i DrivingInCountry_3840x1920_30fps_8bit_420_erp_crf18.mp4
VideoDecoderSpeedTester -dt 0  -nt 10 -sf 20 -nf 200 -i DrivingInCountry_3840x1920_30fps_8bit_420_erp_crf23.mp4

VideoDecoderSpeedTester -dt 1 -sf 20 -nf 200 -i DrivingInCountry_3840x1920_30fps_8bit_420_erp_crf18.mp4

VideoDecoderSpeedTester -dt 1 -sf 20 -nf 200 -i DrivingInCountry_3840x1920_30fps_8bit_420_erp_crf23.mp4

VideoDecoderSpeedTester -dt -1 -sf 20 -nf 200 -w 3840 -h 1920 -bd 8 -i DrivingInCountry_3840x1920_30fps_8bit_420_erp.yuv