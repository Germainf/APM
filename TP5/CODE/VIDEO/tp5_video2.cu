#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>
#include <omp.h>
#include <cuda_runtime.h>
#include "libvideo.h"

#define SEUIL 50

void call_kernel(char * cuda_frame_in, int height, int width);

int main (int argc, char * argv[])
{
	int cpt_frame;
	int frame_count;
	int width, height;

	cuInit(0);

	printf("Opening videos - read and write\n"); fflush(stdout);

	OpenReadAndWriteVideo("./Wildlife.wmv", "./Wildlife_sobel.wmv");

	printf("----------------------------------------\n");
	frame_count = getFrameCount();
	width = getWidth();
	height = getHeight();
	printf("Frame count = %d\n", frame_count); fflush(stdout);

	printf("Width  of frames: %d\n", width); fflush(stdout);
	printf("Height of frames: %d\n", height); fflush(stdout);

	int nb_threads=omp_get_max_threads();
	int nb_gpus;
	cudaGetDeviceCount(&nb_gpus);
	
	printf("nb_threads = %d\n", nb_threads);

	char * frames = (char *) malloc( sizeof(char) * nb_threads * width * height * 3);

 	int limit = (500 / nb_threads) * nb_threads;	
	printf("limit = %d\n", limit);
	int meta_cpt_frame;

	for(meta_cpt_frame = 180; meta_cpt_frame < limit && meta_cpt_frame < frame_count; meta_cpt_frame += nb_threads)
	{
		#pragma omp parallel for
		for(cpt_frame = 0; cpt_frame < nb_threads; cpt_frame++)
		{
			CUcontext myctx;
			cuCtxCreate(&myctx, 0, cpt_frame % nb_gpus); 

			char * cuda_frame_in, * cuda_frame_out;
			cuMemAlloc((CUdeviceptr *)&cuda_frame_in , sizeof(char) * width * height * 3);
			cuMemAlloc((CUdeviceptr *)&cuda_frame_out, sizeof(char) * width * height * 3);

			printf("%d - Read frame with index\n", meta_cpt_frame + cpt_frame); fflush(stdout);
			char * frame1 = &frames[cpt_frame * width * height * 3];
			printf("reading frame %d\n", meta_cpt_frame + cpt_frame); fflush(stdout);

			for(int i=0; i<nb_threads; i++)
			{
				#pragma omp barrier
				if(i==omp_get_thread_num())
				{
					readFrame_with_index(frame1, meta_cpt_frame + cpt_frame);
				}
			}

			if(meta_cpt_frame + cpt_frame > 200 && meta_cpt_frame + cpt_frame < 300)
			{
				printf("%d - GREY\n", cpt_frame); fflush(stdout);

				cuMemcpyHtoD((CUdeviceptr)cuda_frame_in, frame1, sizeof(char) * width * height * 3);
				call_kernel(cuda_frame_in, height, width);
				cuMemcpyDtoH(frame1, (CUdeviceptr)cuda_frame_in, sizeof(char) * width * height * 3);
			}
			cuMemFree((CUdeviceptr) cuda_frame_in);
			cuMemFree((CUdeviceptr) cuda_frame_out);

			#pragma omp ordered
				writeFrame (frame1);
		}
//		printf("First set of threads finished\n");
	}
	
	printf("ECRITURE VIDEO FINIE\n");

    /******************************/
    /**** TP3 - FIN QUESTION 4 ****/
    /******************************/

	free(frames);

	return 0;
}

