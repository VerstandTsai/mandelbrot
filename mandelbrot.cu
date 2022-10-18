#include <SDL2/SDL.h>
#include <cuda.h>
#include <cuComplex.h>
#include <stdio.h>

#define THREAD_COUNT 1024
#define PIXELS_PER_UNIT 512
#define COLOR_DEPTH 4

const int window_width = PIXELS_PER_UNIT * 3;
const int window_height = PIXELS_PER_UNIT * 2;

__global__ void calcscr(int *screen) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int y = tid / window_width;
    int x = tid % window_width;
    const double step = 1.0 / PIXELS_PER_UNIT;
    cuDoubleComplex z = make_cuDoubleComplex(0.0, 0.0);
    cuDoubleComplex c = make_cuDoubleComplex(x*step-2.0, -y*step+1.0);
    int k;
    for (k=0; k<(1<<(COLOR_DEPTH*3)); k++) {
        z = cuCadd(cuCmul(z, z), c);
        if (cuCabs(z) > 2.0) break;
    }
    screen[y*window_width + x] = k;
}

int main(void) {
    SDL_Event event;
    SDL_Renderer *renderer;
    SDL_Window *window;

    SDL_Init(SDL_INIT_VIDEO);
    SDL_CreateWindowAndRenderer(window_width, window_height, 0, &window, &renderer);

    int pixel_num = window_height * window_width;
    int arraylen = sizeof(int) * pixel_num;
    int screen[pixel_num];
    int *g_screen;
    cudaMalloc(&g_screen, arraylen);

    calcscr<<<window_width * window_height / THREAD_COUNT, THREAD_COUNT>>>(g_screen);

    cudaMemcpy(&screen, g_screen, arraylen, cudaMemcpyDeviceToHost);
    cudaFree(g_screen);

    for (int y=0; y<window_height; y++) {
        for (int x=0; x<window_width; x++) {
            int k = screen[y*window_width + x];
            unsigned char r, g, b;
            r = ((k >> (COLOR_DEPTH << 1)) & ((1 << COLOR_DEPTH)-1)) << (8-COLOR_DEPTH);
            g = ((k >> COLOR_DEPTH) & ((1 << COLOR_DEPTH)-1)) << (8-COLOR_DEPTH);
            b = (k & ((1 << COLOR_DEPTH)-1)) << (8-COLOR_DEPTH);
            SDL_SetRenderDrawColor(renderer, r, g, b, 0);
            SDL_RenderDrawPoint(renderer, x, y);
        }
    }
    SDL_RenderPresent(renderer);

    while (1) {
        if (SDL_PollEvent(&event) && event.type == SDL_QUIT) break;
    }

    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();
    return 0;
}
