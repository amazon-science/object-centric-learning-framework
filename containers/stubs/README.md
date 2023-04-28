# Stub files

Some libraries are not delivered with the NVIDIA docker image as they are part
of the NVIDIA driver. The NVIDIA container toolkit dynamically links these
libraries into the container when the container is run.

In order to allow compiling software that relies on these libraries on machines
without NVIDIA driver installation we rely on stub libraries. These are fake
libraries which contain the same symbols as the original library but no
funcitonality. This makes the linker happy at compile time.

For more information see: https://github.com/NVIDIA/nvvl/tree/master/examples/pytorch_superres/docker

To build the container with GPU-based video decoding support, please download
the file `libnvcuvid.so` from the link above or generate it using the
`make-stub.sh` script.
