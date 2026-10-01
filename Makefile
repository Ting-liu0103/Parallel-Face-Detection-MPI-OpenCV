CXX = mpicxx
CXXFLAGS = -O3 -fopenmp -std=c++11

OPENCV_CFLAGS = $(shell pkg-config --cflags opencv4)
OPENCV_LIBS = $(shell pkg-config --libs opencv4)
RPATH = -Wl,-rpath,$(CONDA_PREFIX)/lib

all: serial_face parallel_face

serial_face: serial_face.cpp
	$(CXX) $(CXXFLAGS) -o serial_face serial_face.cpp $(OPENCV_CFLAGS) $(OPENCV_LIBS) $(RPATH)

parallel_face: parallel_face.cpp
	$(CXX) $(CXXFLAGS) -o parallel_face parallel_face.cpp $(OPENCV_CFLAGS) $(OPENCV_LIBS) $(RPATH)

clean:
	rm -f serial_face parallel_face
