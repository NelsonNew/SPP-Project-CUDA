CC=nvcc
CFLAGS=-I. -std=c++11
EXEC=bilateral
SRCS=main.cpp serial.cpp kernel.cu ppm.cpp

all:
	$(CC) $(CFLAGS) $(SRCS) -o $(EXEC)


clean:
	rm -f *.o *~ $(EXEC) image_*