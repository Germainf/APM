all: tp5_video2.pgr 

NVCC=nvcc
CC=g++
CFLAGS= -I./libvideo/ -I. -I/usr/include -L./opencv2/core -L./libvideo/ -lvideo -lopencv_highgui -lopencv_core -L/usr/lib/x86_64-linux-gnu/ -lgomp -lcuda -lcudart
LDFLAGS= -I./libvideo/ -I. -I/usr/include  -L./opencv2/core -L./libvideo/ -lvideo -lopencv_highgui -lopencv_core -L/usr/lib/x86_64-linux-gnu/ -lgomp -lcuda -lcudart
OMPFLAGS= -I./libvideo/ -I. -I/usr/include -L./opencv2/core -L./libvideo/ -lvideo -lopencv_highgui -lopencv_core -L/usr/lib/x86_64-linux-gnu/ -lgomp -lcuda -lcudart -fopenmp



%.pgr: %.o kernel_q2.o
	$(NVCC) $(LDFLAGS) $^ -o $@ 

#tp5_video2.o: tp5_video2.c
#	$(CC) $(OMPFLAGS) $< -c -o $@

%.o: %.cu
	$(NVCC) $(CFLAGS)  $< -c -o $@



clean:
	rm -rf *.o *.pgr
