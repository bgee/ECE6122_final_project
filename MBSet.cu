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
#include <fstream>

#include <GL/freeglut.h>

// Size of window in pixels, both width and height
#define WINDOW_DIM            512
#define DIFF                  3.0
#define THREADS_PER_BLOCK     32

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
RGB *h_results, *d_results, *h_colors, *d_colors;
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


void display(void)
{
  glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
  glClear(GL_COLOR_BUFFER_BIT);
  glLoadIdentity();
  //gluLookAt(0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0);
  // glTranslatef(WINDOW_DIM/2, WINDOW_DIM/2, 0);
  glScalef(0.003, 0.003, 0);
  glBegin(GL_POINTS);
  for (int i = 0; i < WINDOW_DIM; i++){
    for (int j = 0; j < WINDOW_DIM; j++){
      RGB current = h_results[i*WINDOW_DIM+j];
      glColor3d(current.r, current.g, current.b);
      //glColor3d(0.5, 0.0, 0.0);
      //glVertex2f(500, 500);
      //cout << current.r << " " << current.g << " ";
      //cout << current.b << endl;
      glVertex2i(i - WINDOW_DIM/2, j - WINDOW_DIM/2);
      //glVertex2i(-i, -j);
      //glVertex2i(i, -j);
      //glVertex2i(-i, j);
      cout << i-WINDOW_DIM/2 << " " << j - WINDOW_DIM/2 << endl;
    }
  }
  glEnd();
  glFinish();
  glutSwapBuffers();
}

void reshape(int w, int h)
{
  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();
}

/*
void setupSet(Complex* set)
{
  for (int i = 0; i < WINDOW_DIM; i++){
    for (int j = 0; j < WINDOW_DIM; j++){
      *(set+i*WINDOW_DIM + j) = minC + Complex(float(i)/float(WINDOW_DIM)*DIFF,
					       float(j)/float(WINDOW_DIM)*DIFF);
    }
  }
}
*/

__global__ void computeSingle(RGB *d_results, RGB *d_colors)
{
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  int row = index / WINDOW_DIM;
  int col = index % WINDOW_DIM;
  Complex current = Complex(-2.0 + DIFF*double(row)/double(WINDOW_DIM),
			    -1.2 + DIFF*double(col)/double(WINDOW_DIM));
  Complex c = Complex(current);
  int count = -1;
  while ((count < 2000) && (current.magnitude2() < 4)){
    current = current * current + c;
    count++;
  }
  d_results[index].r = d_colors[count].r;
  d_results[index].g = d_colors[count].g;
  d_results[index].b = d_colors[count].b;
}


__global__ void computeSet(RGB *d_results, RGB *d_colors)
{
  int b = WINDOW_DIM;
  for (int i = 0; i < b; i++){
    for (int j = 0; j < b; j++){
      //printf("abc %d, %d\n", i, j);
      int count = -1;
      Complex current = Complex(-2.0 + DIFF*double(i)/double(WINDOW_DIM),
			        -1.2 + DIFF*double(j)/double(WINDOW_DIM));
      Complex c = Complex(current);
      //printf("before while\n");
      while ((count < 2000) && (current.magnitude2() < 4)){
	current = current * current + c;
	count++;
      }
      //printf("after while%d\n", count);
      d_results[i*WINDOW_DIM + j].r = d_colors[count].r;
      d_results[i*WINDOW_DIM + j].g = d_colors[count].g;
      d_results[i*WINDOW_DIM + j].b = d_colors[count].b;
      //printf("%d, %d, %d\n", i, j, count);
    }
  }
  for (int i = 0; i < b; i++){
    for (int j = 0; j < b; j++){
      if (d_results[i*b + j].r != 0.0){
	//printf("%d, %d ", i, j);
      }
    }
  }
  printf("\n");
}


int main(int argc, char** argv)
{
  // debug file to output RGB
  ofstream r, g, b;
  r.open("r.csv");
  g.open("g.csv");
  b.open("b.csv");
  // Initialize OPENGL here
  // Set up necessary host and device buffers
  // set up the opengl callbacks for display, mouse and keyboard
  glutInit(&argc, argv);
  glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH | GLUT_MULTISAMPLE);
  glutInitWindowSize(WINDOW_DIM, WINDOW_DIM);
  glutInitWindowPosition(100, 100);
  glutCreateWindow("MB Set");
  glClearColor(0.0, 0.0, 0.0, 1.0);
  cout << "here" << endl;
  h_colors = colors;
  // allocate space for pixels
  cudaMalloc((void **)&d_results, WINDOW_DIM*WINDOW_DIM*sizeof(RGB));
  cudaMalloc((void **)&d_colors, (maxIt+1)*sizeof(RGB));
  h_results = new RGB[WINDOW_DIM*WINDOW_DIM*sizeof(RGB)];
  cout << "after cudaMalloc" << endl;
  InitializeColors();

  cout << "after colors" << endl;
  cudaMemcpy(d_colors, colors, (maxIt + 1)*sizeof(RGB),
	     cudaMemcpyHostToDevice);
  cout << "after cudaMemcpy" << endl;
  // allocate space for complex matrix
  // set = new Complex[WINDOW_DIM * WINDOW_DIM];
  // Calculate the interation counts
  cout << "before comuteSet" << endl;
  //computeSet<<<1, 1>>>(d_results, d_colors);
  computeSingle<<<WINDOW_DIM*WINDOW_DIM/THREADS_PER_BLOCK,
    THREADS_PER_BLOCK>>>(d_results, d_colors);
  cout << "after computeSet" << endl;
  cudaMemcpy(h_results, d_results, WINDOW_DIM*WINDOW_DIM*sizeof(RGB),
	     cudaMemcpyDeviceToHost);
  cout << "after copy result" << endl;
  for (int i = 0; i < WINDOW_DIM*WINDOW_DIM; i++){
    //h_results[i] = RGB(1, 1, 1);
    //cout << h_results[i].r << " ";
  }
  for (int i = 0; i < WINDOW_DIM; i++){
    for (int j = 0; j < WINDOW_DIM; j++){
      if (h_results[i*WINDOW_DIM + j].r != 0.0){
	cout << i << " " << j << " ";
      }
      RGB c = h_results[i*WINDOW_DIM + j];
      r << c.r << ",";
      g << c.g << ",";
      b << c.b << ",";
    }
    r << "\n"; g << "\n"; b << "\n";
  }
  r.close(); g.close(); b.close();
  cout << endl;
  cudaFree(d_results);
  glutDisplayFunc(display);
  // Grad students, pick the colors for the 0 .. 1999 iteration count pixels

  glutMainLoop(); // THis will callback the display, keyboard and mouse
  return 0;

}
