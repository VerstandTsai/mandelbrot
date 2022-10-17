#include <SDL2/SDL.h>
#include <stdio.h>
#include <complex.h>

#define PIXELS_PER_UNIT 200

int main(void) {
    const int window_width = PIXELS_PER_UNIT * 3;
    const int window_height = PIXELS_PER_UNIT * 2;
    const double step = 1.0 / PIXELS_PER_UNIT;

    SDL_Event event;
    SDL_Renderer *renderer;
    SDL_Window *window;

    SDL_Init(SDL_INIT_VIDEO);
    SDL_CreateWindowAndRenderer(window_width, window_height, 0, &window, &renderer);

    for (int i=0; i<window_height; i++) {
        for (int j=0; j<window_width; j++) {
            double complex z = 0.0 + 0.0*I;
            double complex c = (j*step-2.0) + (-i*step+1.0)*I;
            int k;
            for (k=0; k<(1<<9); k++) {
                if (SDL_PollEvent(&event) && event.type == SDL_QUIT) goto close_window;
                z = z*z + c;
                if (cabs(z) > 2.0) break;
            }
            unsigned char r, g, b;
            r = ((k >> 6) & 7) << 5;
            g = ((k >> 3) & 7) << 5;
            b = (k & 7) << 5;
            SDL_SetRenderDrawColor(renderer, r, g, b, 0);
            SDL_RenderDrawPoint(renderer, j, i);
            printf("\rDrawing pixels: %d/%d", i*window_width + j, window_width*window_height);
        }
    }
    SDL_RenderPresent(renderer);

    while (1) {
        if (SDL_PollEvent(&event) && event.type == SDL_QUIT) break;
    }

    close_window:
    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();
    return 0;
}
