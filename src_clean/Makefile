# Incremental Makefile for gRASPA (Socket patch).
# Uses nvc++ from NVIDIA HPC SDK.  The NVHPC environment variable must be set
# (load the NVIDIA HPC SDK module before calling make).
#
# Usage:
#   make           — compile only changed translation units, then link
#   make clean     — remove object files, dep files, and the binary
#   make cleanobj  — remove object files and dep files only (keeps binary)
#
# NVC_COMPILE is the legacy monolithic rebuild script; this Makefile replaces
# it for everyday development by retaining .o files between builds.

CXX      := nvc++
TARGET   := nvc_main.x
torchDir := /storage/home/hcoda1/9/ltimmerman3/scratch/gRASPA-container/libtorch

CXXFLAGS := -O3 -std=c++20 -target=gpu -Minline -fopenmp -cuda -stdpar=multicore \
            -D_GLIBCXX_USE_CXX11_ABI=1 \
            -I$(torchDir)/include \
            -I$(torchDir)/include/torch/csrc/api/include

LDFLAGS  := -L$(torchDir)/lib \
            -L$(NVHPC)/Linux_x86_64/24.5/cuda/lib64 \
            -L/usr/local/pace-apps/spack/modules/lmod/linux-rhel9-x86_64/gcc/12.3.0 \
            -Wl,-R$(torchDir)/lib \
            -ltorch -ltorch_cpu -lc10 -ltorch_cuda

SRCS_CU  := axpy.cu VDW_Coulomb.cu
SRCS_CPP := main.cpp read_data.cpp data_struct.cpp
OBJS     := $(SRCS_CU:.cu=.o) $(SRCS_CPP:.cpp=.o)
DEPS     := $(OBJS:.o=.d)

.PHONY: all clean cleanobj

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) $^ -o $@

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -MMD -MP -c $< -o $@

%.o: %.cu
	$(CXX) $(CXXFLAGS) -MMD -MP -c $< -o $@

-include $(DEPS)

clean: cleanobj
	rm -f $(TARGET)

cleanobj:
	rm -f $(OBJS) $(DEPS)
