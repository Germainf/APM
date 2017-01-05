#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>
#include "libvideo.h"

#define SEUIL 50

#define RED 0
#define GREEN 1
#define BLUE 2


__global__ void kernel_grey(char * frame, int height, int width)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	
	int red, green, blue;


	for(; i<width * height; i+= ( gridDim.x * blockDim.x))
	{

		red = frame[i*3 + RED];
		green = frame[i*3 + GREEN];
		blue = frame[i*3 + BLUE];

		float moy = 0.3*red + 0.6*green + 0.1*blue;

		frame[i*3 + RED] = (char)moy;
		frame[i*3 + GREEN] = (char)moy;
		frame[i*3 + BLUE] = (char)moy;
	}
}


__global__ void kernel_sobel(char * frame_in, char * frame_out, int height, int width)
{
	int i = threadIdx.x+1;
	int j = blockIdx.x+1;

	float no, n, ne, so, s, se, o, e;
	float deltaX, deltaY;


	for(; j<height-1; j+=gridDim.x)
	{
		for(;i<width-1; i+=blockDim.x)
		{


			no = frame_in[((j+1)*width+(i-1))*3+RED];
			n  = frame_in[((j+1)*width+(i  ))*3+RED];
			ne = frame_in[((j+1)*width+(i+1))*3+RED];
			so = frame_in[((j-1)*width+(i-1))*3+RED];
			s  = frame_in[((j-1)*width+(i  ))*3+RED];
			se = frame_in[((j-1)*width+(i+1))*3+RED];
			o  = frame_in[((j  )*width+(i-1))*3+RED];
			e  = frame_in[((j  )*width+(i+1))*3+RED];

			deltaX = -no+ne-2*o+2*e-so+se;
			deltaY = se+2*s+so-ne-2*n-no;

			float val_red = sqrt(deltaX*deltaX + deltaY*deltaY);


			no = frame_in[((j+1)*width+(i-1))*3+GREEN];
			n  = frame_in[((j+1)*width+(i  ))*3+GREEN];
			ne = frame_in[((j+1)*width+(i+1))*3+GREEN];
			so = frame_in[((j-1)*width+(i-1))*3+GREEN];
			s  = frame_in[((j-1)*width+(i  ))*3+GREEN];
			se = frame_in[((j-1)*width+(i+1))*3+GREEN];
			o  = frame_in[((j  )*width+(i-1))*3+GREEN];
			e  = frame_in[((j  )*width+(i+1))*3+GREEN];

			deltaX = -no+ne-2*o+2*e-so+se;
			deltaY = se+2*s+so-ne-2*n-no;

			float val_green = sqrt(deltaX*deltaX + deltaY*deltaY);


			no = frame_in[((j+1)*width+(i-1))*3+BLUE];
			n  = frame_in[((j+1)*width+(i  ))*3+BLUE];
			ne = frame_in[((j+1)*width+(i+1))*3+BLUE];
			so = frame_in[((j-1)*width+(i-1))*3+BLUE];
			s  = frame_in[((j-1)*width+(i  ))*3+BLUE];
			se = frame_in[((j-1)*width+(i+1))*3+BLUE];
			o  = frame_in[((j  )*width+(i-1))*3+BLUE];
			e  = frame_in[((j  )*width+(i+1))*3+BLUE];

			deltaX = -no+ne-2*o+2*e-so+se;
			deltaY = se+2*s+so-ne-2*n-no;

			float val_blue = sqrt(deltaX*deltaX + deltaY*deltaY);


			float sobel_val = (val_red + val_green + val_blue)/3;

			if(sobel_val > SEUIL)
			{
				frame_out[(j*width+i)*3+RED]   = (char)255;
				frame_out[(j*width+i)*3+GREEN] = (char)255;
				frame_out[(j*width+i)*3+BLUE]  = (char)255;
			}
			else
			{
				frame_out[(j*width+i)*3+RED]   = (char)0;
				frame_out[(j*width+i)*3+GREEN] = (char)0;
				frame_out[(j*width+i)*3+BLUE]  = (char)0;

			}
		}
	}
}




int main (int argc, char * argv[])
{
	int cpt_frame;
	int frame_count;
	int width, height;
	int nGPUs;
	cudaGetDeviceCount(&nGPUs);


	printf("Opening videos - read and write\n"); fflush(stdout);

	OpenReadAndWriteVideo("./Wildlife.wmv", "./Wildlife_sobel.wmv");


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
	for(int i=0 ; i<nGPUs ; i++)
	{
		cudaSetDevice(i);
		cudaMalloc((void **)&cuda_frame_in , sizeof(char) * width * height * 3);
		cudaMalloc((void **)&cuda_frame_out, sizeof(char) * width * height * 3);
	}
	cudaSetDevice(0);

	
	for(cpt_frame = 0; cpt_frame < 500 && cpt_frame < frame_count; cpt_frame ++)
	{

		printf("%d - Read frame with index\n", cpt_frame); fflush(stdout);
		readFrame_with_index(frame1, cpt_frame);


	dd origin git@github.com:Germainf/APM.git	if(cpt_frame > 200 && cpt_frame < 300)
		{
			cudaSetDevice(cpt_frame%nGPUs);
			printf("%d - GREY\n", cpt_frame); fflush(stdout);

			cudaMemcpy(cuda_frame_in, frame1, sizeof(char) * width * height * 3, cudaMemcpyHostToDevice);
			dim3 mygrid;
			mygrid.x =  height/4;
			dim3 myblock;
			myblock.x = width/4;
			kernel_grey<<<mygrid, myblock>>>(cuda_frame_in, height, width);
			cudaMemcpy(frame1, cuda_frame_in, sizeof(char) * width * height * 3, cudaMemcpyDeviceToHost);
		}

		if(cpt_frame >= 300 && cpt_frame < 800)
		{
			
			printf("%d - SOBEL\n", cpt_frame); fflush(stdout);



			cudaMemcpy(cuda_frame_in, frame1, sizeof(char) * width * height * 3, cudaMemcpyHostToDevice);
			dim3 mygrid;
			mygrid.x =  (height)/1;
			dim3 myblock;
			myblock.x = (width)/16;
			kernel_sobel<<<mygrid, myblock>>>(cuda_frame_in, cuda_frame_out, height, width);
			cudaMemcpy(frame1, cuda_frame_out, sizeof(char) * width * height * 3, cudaMemcpyDeviceToHost);


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

