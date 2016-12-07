APPLICATION=benchmark
LIBRARY_DIR = -Ilibrary
LIBRARY = -lLibrary
SOURCES = $(wildcard *.cpp)
OBJECTS = $(SOURCES:.cpp=.o)
ASSEMBLY = $(SOURCES:.cpp=.S)

#CC = g++-4.8
CC = g++
#CC = clang++-3.5
CFLAGS = -Wall -Wextra -O3  -std=c++11 -mavx2 -mfma -fopenmp -g  # -march=native -mtune=native
#CFLAGSPERF= -ftree-vectorize  -Rpass=loop-vectorize -Rpass-missed=loop-vectorize -Rpass-analysis=loop-vectorize
CFLAGSPERF= -ftree-vectorize  -fopt-info-vec #-fopt-info-vec-missed #-ftree-vectorizer-verbose=1
CFLAGSUNSAFE=  -ffast-math -funsafe-math-optimizations
CFLAGSASM= #-fverbose-asm
LDFLAGS = -fopenmp

all: $(OBJECTS) $(ASSEMBLY)
	@echo $(SOURCES)
	g++ -o $(APPLICATION) $(OBJECTS) $(LDFLAGS)

%.S: %.cpp
	$(CC) $(CFLAGS) $(CFLAGSPERF) $(CFLAGSUNSAFE) $(CFLAGSASM) -S -o $@ -c $<

# mult.o: mult.cpp
# 	$(CC) $(CFLAGS) $(CFLAGSPERF) $(CFLAGSUNSAFE) -o $@ -c $<

%.o: %.cpp
	$(CC) $(CFLAGS) $(CFLAGSPERF) $(CFLAGSUNSAFE) -o $@ -c $<

clean:
	-@rm $(APPLICATION)
	-@rm *.o
	-@rm *.S
