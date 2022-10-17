CC = /usr/local/cuda/bin/nvcc
CFLAGS =
LIBS = -lSDL2
SRC = mandelbrot.cu

all: $(SRC)
	$(CC) $(CFLAGS) $(SRC) -o mandelbrot $(LIBS)

clean:
	rm -f *.o mandelbrot

