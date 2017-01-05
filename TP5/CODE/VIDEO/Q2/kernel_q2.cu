#define RED 0
#define GREEN 1
#define BLUE 2

void call_kernel(char * cuda_frame_in, int height, int width);

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

void call_kernel(char * cuda_frame_in, int height, int width)
{
	dim3 mygrid;
	mygrid.x =  height/4;
	dim3 myblock;
	myblock.x = width/4;
	kernel_grey<<<mygrid, myblock>>>(cuda_frame_in, height, width);

}
