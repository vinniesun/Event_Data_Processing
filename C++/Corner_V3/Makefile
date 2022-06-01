CXX = g++

TARGET = main

INCLUDES = -I/usr/local/Cellar/eigen/3.4.0_1/include/eigen3/ -I/usr/local/Cellar/opencv/4.5.5_1/include/opencv4/ -I include
LDFLAGS = -lopencv_core -lopencv_imgcodecs -lopencv_imgproc -lopencv_highgui -lopencv_ml -lopencv_video -lopencv_features2d -lopencv_videoio -lopencv_photo 

# Compiler Flag
CXXFLAGS_o = -std=c++11 -w -O2 $(INCLUDES)
CXXFLAGS = -std=c++11 -w -O2 $(INCLUDES)
CXXFLAGS_DEBUG = -std=c++11 -w -g3 -O2 $(INCLUDES)

INC_DIR = ./include/
SRC = ./src/

# Dependencies
DEPS = $(INC_DIR)efast.hpp $(INC_DIR)arcstar.hpp $(INC_DIR)luvharris.hpp $(INC_DIR)process.hpp $(INC_DIR)customharris.hpp $(INC_DIR)progressbar.hpp
# Objects
OBJ = efast.o arcstar.o luvharris.o process.o customharris.o main.o 

# Here the -c option means to compile or assemble the source files, but do not link.
%.o: $(SRC)%.cpp $(DEPS)
	$(CXX) $(CXXFLAGS_o) -c -o $@ $<

$(TARGET): $(OBJ)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

main_debug: $(OBJ)
	$(CXX) $(CXXFLAGS_DEBUG) -o $@ $^ $(LDFLAGS)

.PHONY: clean
clean:
	rm -f *.o