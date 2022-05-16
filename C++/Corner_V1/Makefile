CXX = g++

TARGET = main

INCLUDES = -I/usr/local/Cellar/eigen/3.4.0_1/include/eigen3/ -I/usr/local/Cellar/opencv/4.5.5_1/include/opencv4/
LDFLAGS = -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_ml -lopencv_video -lopencv_features2d

# Compiler Flag
CXXFLAGS_o = -std=c++11 -w -O2 $(INCLUDES)
CXXFLAGS = -std=c++11 -w -O2 $(INCLUDES)

# Dependencies
DEPS = efast.hpp arcstar.hpp luvharris.hpp process.hpp
# Objects
OBJ = efast.o arcstar.o luvharris.o process.o main.o

# Here the -c option means to compile or assemble the source files, but do not link.
%.o: %.c $(DEPS)
	$(CXX) $(CXXFLAGS_o) -c -o $@ $<

$(TARGET): $(OBJ)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

.PHONY: clean
clean:
	rm -f *.o