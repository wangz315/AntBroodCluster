# name of the executable to be generated
PROG = AntBinCluster
INCLUDE_DIR = includes
SRC_DIR = src
# space-delimited list of header files
HDR_FILES = brood.h globals.h init.h run.h fin.h
# space-delimited list of source files
SRC_FILES = brood.cpp init.cpp run.cpp fin.cpp main.cpp
# space-delimited list of CUDA header files
CUDA_HDRS = 
# space-delimited list of CUDA source files
CUDA_SRCS = 

# Tack on the appropriate dir name (cut down on the amount of typing required above)
C_HDRS = $(patsubst %.h, $(INCLUDE_DIR)/%.h, $(HDR_FILES))
C_SRCS = $(patsubst %.cpp, $(SRC_DIR)/%.cpp, $(SRC_FILES))


# directory to store object files
C_OBJDIR = ./object/c
C_OBJS = $(patsubst %.cpp, $(C_OBJDIR)/%.o, $(C_SRCS))
C_CC = g++
C_CFLAGS = -I$(INCLUDE_DIR) -Wall
C_LIBS =


####################
# Compiling rules: #
####################
# WARNING: *must* have a tab before each definition
all : $(C_OBJDIR) $(PROG)

# links object files into executable
$(PROG) : $(C_OBJS)
	$(C_CC) $(C_CFLAGS) $(subst /src/,/,$^) -o $(PROG) $(C_LIBS)

# compiles source files into object files
$(C_OBJDIR)/%.o : %.cpp $(C_HDRS)
	$(C_CC) -c $(C_CFLAGS) $< -o $(C_OBJDIR)/$(notdir $@) $(C_LIBS)

# creates directory to store object files
$(C_OBJDIR) :
	mkdir -p $@/


####################
# running rules:   #
####################
run :
	./$(PROG)

1 :
	./$(PROG) -f graphs/15nodeSample.gr -d 10 -a 4 -i 500 -k 3

2 :
	./$(PROG) -f graphs/karate.gr -d 20 -a 4 -i 15000 -k 4 -1 0.49 -2 0.06 -l 1 -m 100 -p 0.5


# cleans up object files and executable
# type "make clean" to use
# Note: you can add your own commands to remove other things (e.g. output files)
clean :
	rm -rf object/
	rm -f $(PROG)
	rm -f output.txt
