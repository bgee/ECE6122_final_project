/* 
 * File:   MBSet.cu
 * 
 * Created on June 24, 2012
 * 
 * Purpose:  This program displays Mandelbrot set using the GPU via CUDA and
 * OpenGL immediate mode.
 * 
 */

#include <iostream>
#include <stack>
#include <cuda_runtime_api.h>
#include <stdio.h>
#include "Complex.cu"

#include <GL/freeglut.h>

// Size of window in pixels, both width and height
#define WINDOW_DIM            512

using namespace std;

// Initial screen coordinates, both host and device.
Complex minC(-2.0, -1.2);
Complex maxC(1.0, 1.8);
Complex* dev_minC;
Complex* dev_maxC;
const int maxIt = 2000; // Msximum Iterations

// Define the RGB Class
class RGB
{
public:
  RGB()
    : r(0), g(0), b(0) {}
  RGB(double r0, double g0, double b0)
    : r(r0), g(g0), b(b0) {}
public:
  double r;
  double g;
  double b;
};

RGB* colors = 0; // Array of color values

void InitializeColors()
{
  colors = new RGB[maxIt + 1];
  for (int i = 0; i < maxIt; ++i)
    {
      if (i < 5)
        { // Try this.. just white for small it counts
          colors[i] = RGB(1, 1, 1);
        }
      else
        {
          colors[i] = RGB(drand48(), drand48(), drand48());
        }
    }
  colors[maxIt] = RGB(); // black
}

int main(int argc, char** argv)
{
  // Initialize OPENGL here
  // Set up necessary host and device buffers
  // set up the opengl callbacks for display, mouse and keyboard

  // Calculate the interation counts
  // Grad students, pick the colors for the 0 .. 1999 iteration count pixels

  InitializeColors();
  glutMainLoop(); // THis will callback the display, keyboard and mouse
  return 0;
  
}
