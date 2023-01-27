# Uncomment some options if things don't work
# export CXX=c++; # set this if you want to use a different C++ compiler
export CUDA_HOME=/usr/local/cuda-11.8; # or select the correct cuda version on your system.
pip install -U git+https://github.com/NVIDIA/MinkowskiEngine -v --no-deps \
                          --install-option="--force_cuda"
#                           \ # uncomment the following line if you want to force no cuda installation. force_cuda supercedes cpu_only
#                           --install-option="--cpu_only" \
#                           \ # uncomment the following line to override to openblas, atlas, mkl, blas
#                           --install-option="--blas=openblas" \