all:
	nvcc project.cu
run:
	nvcc project.cu --run
debug:
	nvcc -g -G  project.cu -o performance
nvprof:
	nvprof ./performance
memcheck:
	cuda-memcheck performance
gdb:
	cuda-gdb performance
server:
	nvcc project.cu -rdc=true -O3 -Xptxas -dlcm=ca -arch=sm_60
