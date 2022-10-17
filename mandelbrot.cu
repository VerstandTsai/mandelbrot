#include <SDL2/SDL.h>
#include <cuda.h>
#include <cuComplex.h>
#include <stdio.h>

#define PIXELS_PER_UNIT 300

const int window_width = PIXELS_PER_UNIT * 3;
const int window_height = PIXELS_PER_UNIT * 2;

__global__ void calcscr(int *screen) {
    int i = blockIdx.x;
    int j = threadIdx.x;
    const double step = 1.0 / PIXELS_PER_UNIT;
    cuDoubleComplex z = make_cuDoubleComplex(0.0, 0.0);
    cuDoubleComplex c = make_cuDoubleComplex(j*step-2.0, -i*step+1.0);
    int k;
    for (k=0; k<(1<<9); k++) {
        z = cuCadd(cuCmul(z, z), c);
        if (cuCabs(z) > 2.0) break;
    }
    screen[i*window_width + j] = k;
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

    calcscr<<<window_height, window_width>>>(g_screen);

    cudaMemcpy(&screen, g_screen, arraylen, cudaMemcpyDeviceToHost);
    cudaFree(g_screen);

    for (int i=0; i<window_height; i++) {
        for (int j=0; j<window_width; j++) {
            int k = screen[i*window_width + j];
            unsigned char r, g, b;
            r = ((k >> 6) & 7) << 5;
            g = ((k >> 3) & 7) << 5;
            b = (k & 7) << 5;
            SDL_SetRenderDrawColor(renderer, r, g, b, 0);
            SDL_RenderDrawPoint(renderer, j, i);
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
