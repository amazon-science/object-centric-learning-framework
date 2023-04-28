# Docker Containers for Running CI and Experiments

 - **CPU_Dockerfile:** Image without CUDA and GPU specific dependencies.
 - **GPU_Dockerfile:** Image with CUDA and GPU specific dependencies.
 - **GPU_video_decoding_Dockerfile:** Image with CUDA and GPU based decoding
   using `decord`. Unfortunately, GPU and CPU decoding seem to differ
   significantly, such that we currently do not use this image.
