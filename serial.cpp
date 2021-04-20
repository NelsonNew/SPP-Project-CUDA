//
//  cpu_pipeline.cpp
//
//  Created by Arya Mazaheri on 01/12/2018.
//
//  modified by Yannic Fischler on 01/29/2020
//

#include "ppm.h"

#include <iostream>
#include <time.h>
#include <chrono>
#include <cmath>

using namespace std;

/*********** Gray Scale Filter  *********/
/**
 * Converts a given 24bpp image into 8bpp grayscale using the CPU.
 */
void cpu_grayscale(int width, int height, BYTE *image, BYTE *image_out) {
  for (int h = 0; h < height; ++h) {
    int offset_out = h * width;      // 1 color per pixel
    int offset     = offset_out * 3; // 3 colors per pixel

    for (int w = 0; w < width; ++w) {
      BYTE *pixel = &image[offset + w * 3];
      // Convert to grayscale following the "luminance" model
      image_out[offset_out + w] =
        pixel[0] * 0.2126f + // R
        pixel[1] * 0.7152f + // G
        pixel[2] * 0.0722f;  // B
    }
  }
}


// 1D Gaussian kernel array values of a fixed size (make sure the number > filter size d)
float fGaussian[64];

void cpu_updateGaussian(int filter_radius, double sd) {
	for (int i = 0; i < 2*filter_radius+1; ++i) {
		float x = i - filter_radius;
		fGaussian[i] = expf(-(x*x) / (2*sd*sd));
	}
}


// Gaussian function for range difference
inline double cpu_gaussian(float x, double sigma) {
	return expf(-(powf(x, 2)) / (2 * powf(sigma, 2)));
}

// Bilateral Filter
void cpu_bilateral_filter(BYTE* input, BYTE* output, int width, int height, int r, double sI, double sS) {
	for(int h = 0; h < height; ++h) {
		for(int w = 0; w < width; ++w) {
			double iFiltered = 0;
			double wP = 0;
			// Get the centre pixel value
			unsigned char centrePx = input[h*width+w];
			// Iterate through filter size from centre pixel
			for (int dy = -r; dy <= r; ++dy) {
				int neighborY = h+dy;
				if (neighborY < 0)
          neighborY = 0;
        else if (neighborY >= height)
          neighborY = height - 1;
				for (int dx = -r; dx <= r; ++dx) {
					int neighborX = w+dx;
					if (neighborX < 0)
	          neighborX = 0;
	        else if (neighborX >= width)
	          neighborX = width - 1;
					// Get the current pixel; value
					unsigned char currPx = input[neighborY*width+neighborX];
					// Weight = 1D Gaussian(x_axis) * 1D Gaussian(y_axis) * Gaussian(Range or Intensity difference)
					double w = (fGaussian[dy + r] * fGaussian[dx + r]) * cpu_gaussian(centrePx - currPx, sI);
					iFiltered += w * currPx;
					wP += w;				
				}
			}
			output[h*width + w] = iFiltered / wP;
		}
	}
}


void cpu_pipeline(const Image & input, Image & output, int filter_radius, double sI, double sS) {
	Image gray_img_out(input.cols, input.rows, 1, "P5");

	// Step1; grayscaling
	auto t1 = std::chrono::high_resolution_clock::now();

	cpu_grayscale(input.cols, input.rows, input.pixels, gray_img_out.pixels);

	auto t2 = std::chrono::high_resolution_clock::now();
	cout << "CPU graycaling time: "
    << std::chrono::duration_cast<std::chrono::microseconds>(t2-t1).count()/1000.0
    << " (ms)\n";

	// Step2; update gaussian
	cpu_updateGaussian(filter_radius,sS);
	// Step3; apply bilateral filter
	t1 = std::chrono::high_resolution_clock::now();
	cpu_bilateral_filter(gray_img_out.pixels, output.pixels, input.cols, input.rows, filter_radius, sI, sS);
	t2 = std::chrono::high_resolution_clock::now();
	cout << "CPU Bilateral Filter time: "
    << std::chrono::duration_cast<std::chrono::microseconds>(t2-t1).count()/1000.0
    << " (ms)\n";
}


