#include <iostream>
#include "ppm.h"

using namespace std;

void gpu_pipeline(const Image & input, Image & output, int r,double sI, double sS);
void cpu_pipeline(const Image & input, Image & output, int r,double sI, double sS);

int main(int argc, char *argv[]) {
	//check for arguments
    if(argc != 3){
        cerr << "Usage: bilateral_pipeline <input_image> <filter_radius_(integer)>" << endl;
        exit(0);
    }
    // Reading input args
    char *img_path = argv[1];
    int filter_radius = atoi(argv[2]);

	Image input = readPPM(img_path);
	// Create output Mat
	Image output_serial = Image(input.cols, input.rows, 1, "P5");
	Image output_gpu = Image(input.cols, input.rows, 1, "P5");

	cpu_pipeline(input, output_serial, filter_radius, 75.0, 75.0);
	savePPM(output_serial, "image_serial.ppm");
	cout << endl;
	gpu_pipeline(input, output_gpu, filter_radius, 75.0, 75.0);
	savePPM(output_gpu, "image_gpu.ppm");
}
