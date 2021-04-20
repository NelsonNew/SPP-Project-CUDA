#ifndef PPM_H
#define PPM_H

#include <string>

typedef unsigned char BYTE;

class Image {
public:
  unsigned int cols, rows, chan; // Image resolution
  BYTE *pixels;                  // 1D array of pixels
  std::string ppm_type;

  Image() : cols(0), rows(0), chan(0), pixels(nullptr), ppm_type("P6") { /* empty image */ }

  Image(const unsigned int _cols, const unsigned int _rows, const unsigned int _chan, const std::string &type) :
    cols(_cols), rows(_rows), chan(_chan), ppm_type(type) {
    pixels = new BYTE[rows * cols * chan];
  }

  Image(Image&& old) : Image(old.cols, old.rows, old.chan, old.ppm_type) {
		pixels = old.pixels;
		old.pixels = nullptr;
		old.rows = 0;
		old.cols = 0;
    old.chan = 0;
	}

  const BYTE &operator[](const unsigned int i) const {
    return pixels[i];
  }

  BYTE &operator[](const unsigned int i) {
    return pixels[i];
  }

  ~Image() {
    if (pixels != nullptr)
      delete[] pixels;
  }
};

Image readPPM(const char *filename);

void savePPM(const Image &img, const char *filename);

#endif
