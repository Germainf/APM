#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>
#include "libvideo.h"

#define SEUIL 50
//#define REPEAT_BLUR 200

#define RED 0
#define GREEN 1
#define BLUE 2

__global__ void kernel_blur(char * frame_in, char * frame_out, int height, int width, int n)
{
	//for(int k=0 ; k<n ; k++)
	{
		for(int y=blockIdx.x ; y<height ; y+=gridDim.x)
			for(int x=threadIdx.x ; x<width ; x+=3*blockDim.x)
			{
				if(x==0 || x==height-1 || y==0 || y==(width-1)*3)
					frame_out[x*width*3+y]=frame_in[x*width*3+y];
				else
				{
					frame_out[3*x*width+y+RED]=(frame_in[x*width*3+y+RED]
								+ frame_in[x*width*3+(y+1)+RED]
								+ frame_in[x*width*3+(y-1)+RED]
								+ frame_in[(x+1)*width*3+y+RED]
								+ frame_in[(x-1)*width*3+y+RED]
								)/5;

					frame_out[3*x*width+y+GREEN]=(frame_in[x*width*3+y+GREEN]
								+ frame_in[x*width*3+(y+1)+GREEN]
								+ frame_in[x*width*3+(y-1)+GREEN]
								+ frame_in[(x+1)*width*3+y+GREEN]
								+ frame_in[(x-1)*width*3+y+GREEN]
								)/5;

					frame_out[3*x*width+y+BLUE]=(frame_in[x*width*3+y+BLUE]
								+ frame_in[x*width*3+(y+1)+BLUE]
								+ frame_in[x*width*3+(y-1)+BLUE]
								+ frame_in[(x+1)*width*3+y+BLUE]
								+ frame_in[(x-1)*width*3+y+BLUE]
								)/5;
				}
			}

		char* tmp=frame_in;
		frame_in=frame_out;
		frame_out=tmp;

	}
}

int main (int argc, char * argv[])
{
	int flous=500;
	int cpt_frame;
	int frame_count;
	int width, height;

	printf("Opening videos - read and write\n"); fflush(stdout);

	OpenReadAndWriteVideo("./Wildlife.wmv", "./Wildlife_flou.wmv");

	printf("----------------------------------------\n");
	frame_count = getFrameCount();
	width = getWidth();
	height = getHeight();
	printf("Frame count = %d\n", frame_count); fflush(stdout);

	printf("Width  of frames: %d\n", width); fflush(stdout);
	printf("Height of frames: %d\n", height); fflush(stdout);

//	char * frames = (char *) malloc( sizeof(char) * frame_count * width * height * 3);
	char * frame1 = (char *) malloc( sizeof(char) * width * height * 3);

    /******************************/
    /****   TP3 - QUESTION 4   ****/
    /******************************/

	char * cuda_frame_in, * cuda_frame_out;
	cudaMalloc((void **)&cuda_frame_in, sizeof(char) * width * height * 3);
	cudaMalloc((void **)&cuda_frame_out, sizeof(char) * width * height * 3);
		
	for(cpt_frame = 190; cpt_frame < 500 && cpt_frame < frame_count; cpt_frame ++)
	{

		printf("%d - Read frame with index\n", cpt_frame); fflush(stdout);
		readFrame_with_index(frame1, cpt_frame);

		if(cpt_frame > 200 && cpt_frame < 300)
		{
			
			printf("%d - BLUR\n", cpt_frame); fflush(stdout);

			cudaMemcpy(cuda_frame_in, frame1, sizeof(char) * width * height * 3, cudaMemcpyHostToDevice);
			dim3 mygrid;
			mygrid.x = 1;
			dim3 myblock;
			myblock.x = 1;
			kernel_blur<<<mygrid, myblock>>>(cuda_frame_in, cuda_frame_out, height, width, flous);
			cudaMemcpy(frame1, /*flous%2?cuda_frame_in:*/cuda_frame_out, sizeof(char) * width * height * 3, cudaMemcpyDeviceToHost);

		}

		writeFrame (frame1);

	}
	printf("ECRITURE VIDEO FINIE\n");

	cudaFree(cuda_frame_in);
	cudaFree(cuda_frame_out);
    /******************************/
    /**** TP3 - FIN QUESTION 4 ****/
    /******************************/

	free(frame1);

	return 0;
}

