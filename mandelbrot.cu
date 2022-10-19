#include <SDL2/SDL.h>
#include <cuda.h>
#include <cuComplex.h>
#include <stdio.h>

const int kWindowWidth = 1280;
const int kWindowHeight = 720;
const double kAspectRatio = (double)kWindowWidth / (double)kWindowHeight;
const double kZoomScale = 0.5;
const int kColorDepth = 4;
const int kNumThreads = 1024;

__global__ void calcscr(int *screen, double center_x, double center_y, double y_range) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int y = tid / kWindowWidth;
    int x = tid % kWindowWidth;
    double pixel_size = y_range / kWindowHeight;
    double cx = x*pixel_size+(center_x-y_range*kAspectRatio/2.0);
    double cy = -y*pixel_size+(center_y+y_range/2.0);
    cuDoubleComplex z = make_cuDoubleComplex(0.0, 0.0);
    cuDoubleComplex c = make_cuDoubleComplex(cx, cy);
    int k;
    for (k=0; k<(1<<(kColorDepth*3)); k++) {
        z = cuCadd(cuCmul(z, z), c);
        if (cuCabs(z) > 2.0) break;
    }
    screen[y*kWindowWidth + x] = k;
}

int main(void) {
    SDL_Event event;
    SDL_Renderer *renderer;
    SDL_Window *window;

    SDL_Init(SDL_INIT_VIDEO);
    SDL_CreateWindowAndRenderer(kWindowWidth, kWindowHeight, 0, &window, &renderer);

    int num_pixels = kWindowWidth * kWindowHeight;
    int arraylen = sizeof(int) * num_pixels;
    int screen[num_pixels];
    int *g_screen;
    cudaMalloc(&g_screen, arraylen);

    calcscr<<<num_pixels / kNumThreads, kNumThreads>>>(g_screen, -0.5, 0.0, 2.0);

    cudaMemcpy(&screen, g_screen, arraylen, cudaMemcpyDeviceToHost);

    for (int y=0; y<kWindowHeight; y++) {
        for (int x=0; x<kWindowWidth; x++) {
            int k = screen[y*kWindowWidth + x];
            unsigned char r, g, b;
            r = ((k >> (kColorDepth << 1)) & ((1 << kColorDepth)-1)) << (8-kColorDepth);
            g = ((k >> kColorDepth) & ((1 << kColorDepth)-1)) << (8-kColorDepth);
            b = (k & ((1 << kColorDepth)-1)) << (8-kColorDepth);
            SDL_SetRenderDrawColor(renderer, r, g, b, 0);
            SDL_RenderDrawPoint(renderer, x, y);
        }
    }
    SDL_RenderPresent(renderer);

    while (1) {
        if (SDL_PollEvent(&event) && event.type == SDL_QUIT) break;
    }

    cudaFree(g_screen);

    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();
    return 0;
}
