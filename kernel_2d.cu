//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Copyright 2021 Brian Hamilton                                                                                    //
//                                                                                                                  //
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated     //
// documentation files (the "Software"), to deal in the Software without restriction, including without             //
// limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of        //
// the Software, and to permit persons to whom the Software is furnished to do so, subject to the following         //
// conditions:                                                                                                      //
//                                                                                                                  //
// The above copyright notice and this permission notice shall be included in all copies or substantial             //
// portions of the Software.                                                                                        //
//                                                                                                                  //
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT            //
// LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO        //
// EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN     //
// AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE     //
// OR OTHER DEALINGS IN THE SOFTWARE.                                                                               //
//                                                                                                                  //
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//                                                                                                                  //
// FDTD tutorial for 180th ASA meeting - CUDA Kernels to accompany Matlab code                                      //
//                                                                                                                  //
// Compiles to PTX from Matlab, but can be compiled to PTX with 'nvcc --ptx kernel_2d.cu'                           //
//                                                                                                                  //
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//
//air update
__global__ void air_update(double *u0, const double * __restrict__ u1, const double * __restrict__ u2, int Nx, int  Ny, bool * in_mask)
{
   int ix = blockIdx.x*blockDim.x + threadIdx.x;
   int iy = blockIdx.y*blockDim.y + threadIdx.y;
   if ((ix>0) && (ix<Nx-1) && (iy>0) && (iy<Ny-1)) {
      int ii = iy*Nx+ix; 
      u0[ii] = (0.5*(u1[ii-1]+u1[ii+1]+u1[ii-Nx]+u1[ii+Nx]) - u2[ii])*in_mask[ii];
   }
}
//rigid boundary update
__global__ void rigid_update(double *u0, const double * __restrict__ u1, const double * __restrict__ u2, int Nx, int Nb, int * ib, int * Kib)
{
   int ix = blockIdx.x*blockDim.x + threadIdx.x;
   if (ix<Nb) {
      int ii = ib[ix]-1; //from matlab indices 
      u0[ii] = (2-0.5*Kib[ix])*u1[ii] + 0.5*(u1[ii-1]+u1[ii+1]+u1[ii-Nx]+u1[ii+Nx]) - u2[ii];
   }
}
//add loss to boundary nodes
__global__ void apply_loss(double *u0, const double * __restrict__ u2, int Nx, int Nb, int * ib, int * Kib, double lf)
{
   int ix = blockIdx.x*blockDim.x + threadIdx.x;
   if (ix<Nb) {
      int ii = ib[ix]-1; //from matlab indices 
      u0[ii] = (u0[ii] + lf*(4-Kib[ix])*u2[ii])/(1.0+lf*(4-Kib[ix]));
   }
}
