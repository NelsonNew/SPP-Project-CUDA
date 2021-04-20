//
//  kernel.cu
//
//  Created by Arya Mazaheri on 01/12/2018.
//

#include <iostream>
#include <algorithm>
#include <cmath>
#include "ppm.h"

using namespace std;

/*********** Gray Scale Filter  *********/

/**
 * Converts a given 24bpp image into 8bpp grayscale using the GPU.
 */
__global__
void cuda_grayscale(int width, int height, BYTE *image, BYTE *image_out)
{

    //TODO: implement grayscale filter kernel
	int tx = blockIdx.x * blockDim.x + threadIdx.x;
	int ty = blockIdx.y * blockDim.y + threadIdx.y;

	int offset_out = ty * width;	// 1 color per pixel	
	int offset = offset_out * 3;	// 3 colors per pixel

	if (tx >= width || ty >= height) return;

	BYTE *pixel = &image[offset + tx * 3];					
	// Convert to grayscale following the "luminance" model
	image_out[offset_out + tx] =							
        pixel[0] * 0.2126f + // R
        pixel[1] * 0.7152f + // G
		pixel[2] * 0.0722f;  // B
}

// 1D Gaussian kernel array values of a fixed size (make sure the number > filter size d)
//TODO: Define the cGaussian array on the constant memory 
__constant__
float cGaussian[64];

void cuda_updateGaussian(int r, double sd)
{
	float fGaussian[64];
	for (int i = 0; i < 2*r +1 ; i++)
	{
		float x = i - r;
		fGaussian[i] = expf(-(x*x) / (2 * sd*sd));
	}
	//TODO: Copy computed fGaussian to the cGaussian on device memory
	
	cudaMemcpyToSymbol(cGaussian, fGaussian, sizeof(float)*64, 0, cudaMemcpyHostToDevice);
}

//TODO: implement cuda_gaussian() kernel
// Gaussian function for range difference
__device__ double cuda_gaussian(float x, double sigma) {
	return expf(-(powf(x, 2)) / (2 * powf(sigma, 2)));
}


/*********** Bilateral Filter  *********/
// Parallel (GPU) Bilateral filter kernel
__global__ void cuda_bilateral_filter(BYTE* input, BYTE* output,
	int width, int height,
	int r, double sI, double sS)
{
	//TODO: implement bilateral filter kernel
	int tx = blockIdx.x * blockDim.x + threadIdx.x;
	int ty = blockIdx.y * blockDim.y + threadIdx.y;
  
	if (tx >= width || ty >= height) return;
  
	double iFiltered = 0;
	double wP = 0;
	unsigned char centrePx = input[ty * width + tx];
  
	for (int dy = -r; dy <= r; dy++) {
	  int neighborY = ty + dy;
	  if (neighborY < 0) 
		neighborY = 0;
	  else if (neighborY >= height) 
		neighborY = height - 1;
	  for (int dx = -r; dx <= r; dx++) {
		int neighborX = tx + dx;
		if (neighborX < 0) 
		  neighborX = 0;
		else if (neighborX >= width) 
		  neighborX = width - 1;
		
		unsigned char currPx = input[neighborY * width + neighborX];
		double w = (cGaussian[dy + r] * cGaussian[dx + r]) * cuda_gaussian(centrePx - currPx, sI);
		iFiltered += w * currPx;
		wP += w;
	  }
	}
	output[ty * width + tx] = iFiltered / wP;
}


void gpu_pipeline(const Image & input, Image & output, int r, double sI, double sS)
{
	// Events to calculate gpu run time
	cudaEvent_t start, stop;
	float time;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	// GPU related variables
	BYTE *d_input = NULL;
	BYTE *d_image_out[2] = {0}; //temporary output buffers on gpu device
	int image_size = input.cols*input.rows;
	int suggested_blockSize;   // The launch configurator returned block size 
	int suggested_minGridSize; // The minimum grid size needed to achieve the maximum occupancy for a full device launch

	// ******* Grayscale kernel launch *************

	//Creating the block size for grayscaling kernel
	cudaOccupancyMaxPotentialBlockSize(&suggested_minGridSize, &suggested_blockSize, cuda_grayscale, 0, image_size);
        
	int block_dim_x, block_dim_y;
	block_dim_x = (int) sqrt(suggested_blockSize);
	block_dim_y = (int) sqrt(suggested_blockSize);

	dim3 gray_block(block_dim_x, block_dim_y);

	//TODO: Calculate grid size to cover the whole image
	int grid_dim_x, grid_dim_y;
  	grid_dim_x = input.cols / block_dim_x;
  	grid_dim_y = input.rows / block_dim_y + 1;

	dim3 gray_grid(grid_dim_x, grid_dim_y);

	// Allocate the intermediate image buffers for each step
	Image img_out(input.cols, input.rows, 1, "P5");
	for (int i = 0; i < 2; i++)
	{  
		//TODO: allocate memory on the device
		cudaMalloc((void**) &d_image_out[i], image_size);
		//TODO: intialize allocated memory on device to zero
		cudaMemset(d_image_out[i], 0, image_size);
	}

	//copy input image to device
	//TODO: Allocate memory on device for input image
	cudaMalloc((void**) &d_input, image_size*3);
	//TODO: Copy input image into the device memory
	cudaMemcpy(d_input, input.pixels, image_size*3,cudaMemcpyHostToDevice);

	cudaEventRecord(start, 0); // start timer
	// Convert input image to grayscale
	//TODO: Launch cuda_grayscale()
	cuda_grayscale<<<gray_grid, gray_block>>>(input.cols, input.rows, d_input, d_image_out[0]);
	cudaEventRecord(stop, 0); // stop timer
	cudaEventSynchronize(stop);

	// Calculate and print kernel run time
	cudaEventElapsedTime(&time, start, stop);
	cout << "GPU Grayscaling time: " << time << " (ms)\n";
	cout << "Launched blocks of size " << gray_block.x * gray_block.y << endl;
    
	//TODO: transfer image from device to the main memory for saving onto the disk
	cudaMemcpy(img_out.pixels, d_image_out[0], image_size,cudaMemcpyDeviceToHost);
	savePPM(img_out, "image_gpu_gray.ppm");
	

	// ******* Bilateral filter kernel launch *************
	
	//Creating the block size for grayscaling kernel
	cudaOccupancyMaxPotentialBlockSize(&suggested_minGridSize, &suggested_blockSize, cuda_bilateral_filter, 0, image_size);

	block_dim_x = (int) sqrt(suggested_blockSize);
	block_dim_y = (int) sqrt(suggested_blockSize);

	dim3 bilateral_block(block_dim_x, block_dim_y);

	//TODO: Calculate grid size to cover the whole image
	
  	grid_dim_x = input.cols / block_dim_x;
	grid_dim_y = input.rows / block_dim_y + 1;
	  
	dim3 bilateral_grid(grid_dim_x, grid_dim_y);

	// Create gaussain 1d array
	cuda_updateGaussian(r,sS);

	cudaEventRecord(start, 0); // start timer
	//TODO: Launch cuda_bilateral_filter() 
	cuda_bilateral_filter<<<bilateral_grid, bilateral_block>>>(d_image_out[0], d_image_out[1], input.cols, input.rows, r, sI, sS);
	cudaEventRecord(stop, 0); // stop timer
	cudaEventSynchronize(stop);

	// Calculate and print kernel run time
	cudaEventElapsedTime(&time, start, stop);
	cout << "GPU Bilateral Filter time: " << time << " (ms)\n";
	cout << "Launched blocks of size " << bilateral_block.x * bilateral_block.y << endl;

	// Copy output from device to host
	//TODO: transfer image from device to the main memory for saving onto the disk 
	cudaMemcpy(output.pixels, d_image_out[1], image_size,cudaMemcpyDeviceToHost);

	// ************** Finalization, cleaning up ************

	// Free GPU variables
	//TODO: Free device allocated memory
	cudaFree(d_image_out[1]);
	cudaFree(d_image_out[0]);
	cudaFree(d_input);
}
