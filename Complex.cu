/* 
 * File:   Complex.cu
 * 
 * Created on June 24, 2012
 * 
 * Purpose:  Simple complex number class for use on GPU
 * 
 * If it works, it was written by Brian Swenson.
 * Otherwise, I have no idea who wrote it.
 */

class Complex 
{
public:
    float   r;
    float   i;
    __host__ __device__ Complex( float a, float b ) : r(a), i(b)  {}
    __device__ Complex(const Complex& x) : r(x.r), i(x.i) {}
    __device__ float magnitude2( void ) {
        return r * r + i * i;
    }
    __device__ Complex operator*(const Complex& a) {
        return Complex(r*a.r - i*a.i, i*a.r + r*a.i);
    }
    __device__ Complex operator+(const Complex& a) {
        return Complex(r+a.r, i+a.i);
    }
};