# name of the executable to be generated
PROG = PABC
INCLUDE_DIR = includes
SRC_DIR = src
# space-delimited list of header files
HDR_FILES = structs.h global.h abc.h init.h run.h fin.h
# space-delimited list of source files
SRC_FILES = abc.cpp init.cpp run.cpp run_mcc.cpp run_mcd.cpp run_mfd.cpp run_mfc.cpp run_pcc.cpp run_pcd.cpp run_pfd.cpp run_pfc.cpp fin.cpp main.cpp
# space-delimited list of CUDA header files
CUDA_HDRS = 
# space-delimited list of CUDA source files
CUDA_SRCS = 

# Tack on the appropriate dir name (cut down on the amount of typing required above)
CU_HDRS = $(patsubst %.h, $(INCLUDE_DIR)/%.h, $(CUDA_HDRS))
CU_SRCS = $(patsubst %.cu, $(SRC_DIR)/%.cu, $(CUDA_SRCS))
C_HDRS = $(patsubst %.h, $(INCLUDE_DIR)/%.h, $(HDR_FILES))
C_SRCS = $(patsubst %.cpp, $(SRC_DIR)/%.cpp, $(SRC_FILES))


# directory to store object files
CUDA_OBJDIR = ./object/cuda
CUDA_OBJS = $(patsubst %.cu, $(CUDA_OBJDIR)/%.o, $(CU_SRCS))
CUDA_CC = nvcc
CUDA_CFLAGS = -L. -I$(INCLUDE_DIR) -O2 -G -g
CUDA_LIBS = -lm

C_OBJDIR = ./object/c
C_OBJS = $(patsubst %.cpp, $(C_OBJDIR)/%.o, $(C_SRCS))
C_CC = g++
C_CFLAGS = -I$(INCLUDE_DIR) -Wall -fopenmp
C_LIBS = -lm


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

$(C_OBJDIR)/%.o : %.cpp $(C_HDRS)
	$(C_CC) -c $(C_CFLAGS) $< -o $(C_OBJDIR)/$(notdir $@) $(C_LIBS)

# creates directory to store object files
$(CUDA_OBJDIR) :
	mkdir -p $@/

$(C_OBJDIR) :
	mkdir -p $@/


####################
# running rules:   #
####################
run :
	./$(PROG)
	./$(PROG) mcc
	./$(PROG) mcd
	./$(PROG) mfc
	./$(PROG) mfd
	./$(PROG) pcc
	./$(PROG) pcd
	./$(PROG) pfc
	./$(PROG) pfd




# cleans up object files and executable
# type "make clean" to use
# Note: you can add your own commands to remove other things (e.g. output files)
clean :
	rm -rf object/
	rm -f $(PROG)
	rm -f output.txt
