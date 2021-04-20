#include "ppm.h"

#include <fstream>
#include <cstring>

Image readPPM(const char *filename) {
  std::ifstream ifs;
  ifs.open(filename, std::ios::binary);
  Image src;
  try {
    if (ifs.fail()) {
      throw("Can't open input file");
      exit(1);
    }

    std::string header;
    int w, h, b;
    ifs >> header;
    if (strcmp(header.c_str(), "P6") != 0)
      throw("Can't read input file");
    ifs >> w >> h >> b;
    if (b != 255)
      throw("Can't read input file");

    src.cols = w;
    src.rows = h;
    src.chan = 3;

    // skip empty lines if necessary until we get to the binary data
    ifs.ignore(256, '\n');

    src.pixels = new BYTE[w * h * 3];

    ifs.read(reinterpret_cast<char *>(src.pixels), w * h * 3);

    ifs.close();
  }
  catch (const char *err) {
    fprintf(stderr, "%s\n", err);
    ifs.close();
  }

  return src;
}

void savePPM(const Image &img, const char *filename) {
  if (img.cols == 0 || img.rows == 0) {
    fprintf(stderr, "Can't save an empty image\n");
    return;
  }

  std::ofstream ofs;
  ofs.open(filename, std::ios::binary); // need to spec. binary mode for Windows users
  try {
    if (strcmp(img.ppm_type.c_str(), "P6") != 0 && strcmp(img.ppm_type.c_str(), "P5") != 0)
      throw("Can't write image to file");

    if (ofs.fail())
      throw("Can't open output file");
    ofs << img.ppm_type << "\n"
        << img.cols << " " << img.rows
        << "\n255\n";

    ofs.write(reinterpret_cast<char *>(img.pixels), img.cols * img.rows * img.chan);
    ofs.close();
  }
  catch (const char *err) {
    fprintf(stderr, "%s\n", err);
    ofs.close();
  }
}
