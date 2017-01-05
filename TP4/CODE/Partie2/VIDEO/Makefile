all: tp4_video1.pgr 

CC=nvcc
CFLAGS= -I./libvideo/ -I. -I/usr/include 
LDFLAGS= -L./opencv2/core -L./libvideo/ -lvideo -lopencv_highgui -lopencv_core -lm




%.pgr: %.o
	$(CC) $(CFLAGS) $< -o $@  $(LDFLAGS)

%.o: %.cu
	$(CC) $(CFLAGS)  $< -c -o $@


clean:
	rm -rf *.o *.pgr
