CC = gcc
CFLAGS = -Wall -O3
LIBS = -lm -lSDL2

all: mandelbrot.o
	$(CC) mandelbrot.o -o mandelbrot $(LIBS)

mandelbrot.o: mandelbrot.c
	$(CC) $(CFLAGS) -c mandelbrot.c

clear:
	rm -f *.o mandelbrot

