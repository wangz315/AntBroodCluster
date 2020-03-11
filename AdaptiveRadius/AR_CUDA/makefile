# name of the executable to be generated
PROG = GPU_AntBroodClustering
INCLUDE_DIR = include
SRC_DIR = src
# space-delimited list of header files
HDR_FILES =
# space-delimited list of source files
SRC_FILES =
# space-delimited list of CUDA header files
CUDA_HDRS = structs.h kernels.h timer.h
# space-delimited list of CUDA source files
CUDA_SRCS = hostRun.cu kernel.cu timer.cu

# Tack on the appropriate dir name (cut down on the amount of typing required above)
CU_HDRS = $(patsubst %.h, $(INCLUDE_DIR)/%.h, $(CUDA_HDRS))
CU_SRCS = $(patsubst %.cu, $(SRC_DIR)/%.cu, $(CUDA_SRCS))
C_HDRS = $(patsubst %.h, $(INCLUDE_DIR)/%.h, $(HDR_FILES))
C_SRCS = $(patsubst %.c, $(SRC_DIR)/%.c, $(SRC_FILES))


# directory to store object files
CUDA_OBJDIR = ./object/cuda
CUDA_OBJS = $(patsubst %.cu, $(CUDA_OBJDIR)/%.o, $(CU_SRCS))
CUDA_CC = nvcc
CUDA_CFLAGS = -arch=sm_13 -Wno-deprecated-gpu-targets -L. -I$(INCLUDE_DIR) -O2 -G -g
CUDA_LIBS = -lm

C_OBJDIR = ./object/c
C_OBJS = $(patsubst %.c, $(C_OBJDIR)/%.o, $(C_SRCS))
C_CC = g++
C_CFLAGS = -L/opt/cuda-6.5/lib64 -O2 -I$(INCLUDE_DIR) -Wall
C_LIBS = -lm -lcudart


####################
# Compiling rules: #
####################
# WARNING: *must* have a tab before each definition
all : $(CUDA_OBJDIR) $(C_OBJDIR) $(PROG)

# links object files into executable
$(PROG) : $(CUDA_OBJS) $(C_OBJS)
	$(C_CC) $(C_CFLAGS) $(subst /src/,/,$^) -o $(PROG) $(C_LIBS)

# compiles source files into object files
$(CUDA_OBJDIR)/%.o : %.cu $(CU_HDRS)
	$(CUDA_CC) -c $(CUDA_CFLAGS) $< -o $(CUDA_OBJDIR)/$(notdir $@) $(CUDA_LIBS)

$(C_OBJDIR)/%.o : %.c $(C_HDRS)
	$(C_CC) -c $(C_CFLAGS) $< -o $(C_OBJDIR)/$(notdir $@) $(C_LIBS)

# creates directory to store object files
$(CUDA_OBJDIR) :
	mkdir -p $@/

$(C_OBJDIR) :
	mkdir -p $@/


####################
# running rules: #
####################
run:
	./$(PROG)

mc:
	cuda-memcheck ./$(PROG)

# cleans up object files and executable
# type "make clean" to use
# Note: you can add your own commands to remove other things (e.g. output files)
clean:
	rm -rf object/
	rm -f $(PROG)
	rm -f output.txt
