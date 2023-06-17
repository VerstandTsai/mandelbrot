#include <SDL2/SDL.h>
#include <cuda.h>
#include <cuComplex.h>
#include <stdio.h>

const int kWindowWidth = 1280;
const int kWindowHeight = 720;
const double kAspectRatio = (double)kWindowWidth / (double)kWindowHeight;
const double kZoomScale = 0.5;
const int kColorDepth = 3;
const int kNumThreads = 1024;

__host__ __device__ void pixel_to_num(
        int pixel_x, int pixel_y,
        double center_x, double center_y,
        double y_range,
        double *num_x, double *num_y
) {
    double pixel_size = y_range / kWindowHeight;
    *num_x = pixel_x*pixel_size + center_x - y_range*kAspectRatio/2.0;
    *num_y = -pixel_y*pixel_size + center_y + y_range/2.0;
}

__global__ void calcscr(int *screen, double center_x, double center_y, double y_range) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int y = tid / kWindowWidth;
    int x = tid % kWindowWidth;
    double cx, cy;
    pixel_to_num(x, y, center_x, center_y, y_range, &cx, &cy);
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
    int num_blocks = num_pixels / kNumThreads;
    int arraylen = sizeof(int) * num_pixels;
    int screen[num_pixels];
    int *g_screen;
    cudaMalloc(&g_screen, arraylen);

    int mouse_x = 0;
    int mouse_y = 0;
    double center_x = -0.5;
    double center_y = 0.0;
    double y_range = 3.0;

    int running = 1;
    while (running) {
        calcscr<<<num_blocks, kNumThreads>>>(g_screen, center_x, center_y, y_range);

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

        while (SDL_PollEvent(&event)) {
            switch (event.type) {
                case SDL_QUIT:
                    running = 0;
                    break;
                case SDL_MOUSEMOTION:
                    mouse_x = event.motion.x;
                    mouse_y = event.motion.y;
                    break;
                case SDL_MOUSEWHEEL:
                    pixel_to_num(mouse_x, mouse_y, center_x, center_y, y_range, &center_x, &center_y);
                    y_range *= (event.wheel.y > 0 ? kZoomScale : 1/kZoomScale);
                    break;
            }
        }
    }

    cudaFree(g_screen);

    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();
    return 0;
}
