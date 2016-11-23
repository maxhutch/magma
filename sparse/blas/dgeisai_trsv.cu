/*
    -- MAGMA (version 2.2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date November 2016

       @generated from sparse/blas/zgeisai_trsv.cu, normal z -> d, Sun Nov 20 20:20:43 2016

*/
#include "magmasparse_internal.h"
//#include <cuda_profiler_api.h>

#define PRECISION_d
#define REAL
#define BLOCKSIZE 256
#define WARP_SIZE 32
#define WRP 32
#define WRQ 1



#include <cuda.h>  // for CUDA_VERSION

#if (CUDA_VERSION >= 7000)
#if (CUDA_ARCH >= 300)

__device__
void dtrsv_lower_kernel_general(double *dA, double *dB, int *sizes)
{
#ifdef REAL
    int j = blockIdx.y * gridDim.x + blockIdx.x;
    int idn = threadIdx.x;

    double rB[ 2 ];
    double rA[ 2 ];

    int n;
    int k;
    int N = sizes[j];

    dA += (j)*WARP_SIZE*WARP_SIZE;
    dB += (j)*WARP_SIZE;


    // Read B to regs.
    #pragma unroll
    for (n = 0; n < 2; n++)
        rB[n] = dB[n*WARP_SIZE+idn];


    // Triangular solve in regs.
    #pragma unroll
    for (k = 0; k < N; k++)
    {
        #pragma unroll
        for (n = 0; n < 2; n++)
            rA[n] = dA[k*WARP_SIZE+n*WARP_SIZE+idn];
        if (k%WARP_SIZE == idn)
            rB[k/WARP_SIZE] /= rA[k/WARP_SIZE];

        double top = __shfl(rB[k/WARP_SIZE], k%WARP_SIZE);

        #pragma unroll
        for (n = 0; n < 2; n++)
            if (n*WARP_SIZE+idn > k)
                rB[n] -= (top*rA[n]);
    }
    // Drop B to dev mem.
    #pragma unroll
    for (n = 0; n < 2; n++)
        if (n*WARP_SIZE+idn < N)
            dB[n*WARP_SIZE+idn] = rB[n];

#endif
}


__device__
void dtrsv_upper_kernel_general(double *dA, double *dB, int *sizes)
{
#ifdef REAL
    int j = blockIdx.y * gridDim.x + blockIdx.x;
    int idn = threadIdx.x;

    double rB[ 2 ];
    double rA[ 2 ];

    int n;
    int k;
    int N = sizes[j];

    dA += (j)*WARP_SIZE*WARP_SIZE;
    dB += (j)*WARP_SIZE;


    // Read B to regs.
    #pragma unroll
    for (n = 0; n < 2; n++)
        rB[n] = dB[n*WARP_SIZE+idn];


    // Triangular solve in regs.
    #pragma unroll
    for (int k = N-1; k > -1; k--)
    {
        #pragma unroll
        for (n = 0; n < 2; n++)
            rA[n] = dA[k*WARP_SIZE+n*WARP_SIZE+idn];
        if (k%WARP_SIZE == idn)
            rB[k/WARP_SIZE] /= rA[k/WARP_SIZE];

        double top = __shfl(rB[k/WARP_SIZE], k%WARP_SIZE);

        #pragma unroll
        for (n = 0; n < 2; n++)
            if (n*WARP_SIZE+idn < k)
                rB[n] -= (top*rA[n]);
    }
    // Drop B to dev mem.
    #pragma unroll
    for (n = 0; n < 2; n++)
        if (n*WARP_SIZE+idn < N)
            dB[n*WARP_SIZE+idn] = rB[n];

#endif
}



__device__
void dtrsv_lower_kernel_1(double *dA, double *dB )
{
#ifdef REAL
    int j = blockIdx.y * gridDim.x + blockIdx.x;
    int idn = threadIdx.x;
    double rB;
    double rA;
    dA += (j)*WARP_SIZE*WARP_SIZE;
    dB += (j)*WARP_SIZE;

    // Read B to regs.
    rB = dB[idn];
    // Triangular solve in regs.
    #pragma unroll
    for (int k = 0; k < 1; k++)
    {
        rA = dA[k*WARP_SIZE+idn];
        if (k%WARP_SIZE == idn)
            rB /= rA;
        double top = __shfl(rB, k%WARP_SIZE);
        if ( idn > k)
            rB -= (top*rA);
    }
    // Drop B to dev mem.
    dB[idn] = rB;
#endif
}



__device__
void dtrsv_lower_kernel_2(double *dA, double *dB )
{
#ifdef REAL
    int j = blockIdx.y * gridDim.x + blockIdx.x;
    int idn = threadIdx.x;
    double rB;
    double rA;
    dA += (j)*WARP_SIZE*WARP_SIZE;
    dB += (j)*WARP_SIZE;

    // Read B to regs.
    rB = dB[idn];
    // Triangular solve in regs.
    #pragma unroll
    for (int k = 0; k < 2; k++)
    {
        rA = dA[k*WARP_SIZE+idn];
        if (k%WARP_SIZE == idn)
            rB /= rA;
        double top = __shfl(rB, k%WARP_SIZE);
        if ( idn > k)
            rB -= (top*rA);
    }
    // Drop B to dev mem.
    dB[idn] = rB;
#endif
}



__device__
void dtrsv_lower_kernel_3(double *dA, double *dB )
{
#ifdef REAL
    int j = blockIdx.y * gridDim.x + blockIdx.x;
    int idn = threadIdx.x;
    double rB;
    double rA;
    dA += (j)*WARP_SIZE*WARP_SIZE;
    dB += (j)*WARP_SIZE;

    // Read B to regs.
    rB = dB[idn];
    // Triangular solve in regs.
    #pragma unroll
    for (int k = 0; k < 3; k++)
    {
        rA = dA[k*WARP_SIZE+idn];
        if (k%WARP_SIZE == idn)
            rB /= rA;
        double top = __shfl(rB, k%WARP_SIZE);
        if ( idn > k)
            rB -= (top*rA);
    }
    // Drop B to dev mem.
    dB[idn] = rB;
#endif
}



__device__
void dtrsv_lower_kernel_4(double *dA, double *dB )
{
#ifdef REAL
    int j = blockIdx.y * gridDim.x + blockIdx.x;
    int idn = threadIdx.x;
    double rB;
    double rA;
    dA += (j)*WARP_SIZE*WARP_SIZE;
    dB += (j)*WARP_SIZE;

    // Read B to regs.
    rB = dB[idn];
    // Triangular solve in regs.
    #pragma unroll
    for (int k = 0; k < 4; k++)
    {
        rA = dA[k*WARP_SIZE+idn];
        if (k%WARP_SIZE == idn)
            rB /= rA;
        double top = __shfl(rB, k%WARP_SIZE);
        if ( idn > k)
            rB -= (top*rA);
    }
    // Drop B to dev mem.
    dB[idn] = rB;
#endif
}



__device__
void dtrsv_lower_kernel_5(double *dA, double *dB )
{
#ifdef REAL
    int j = blockIdx.y * gridDim.x + blockIdx.x;
    int idn = threadIdx.x;
    double rB;
    double rA;
    dA += (j)*WARP_SIZE*WARP_SIZE;
    dB += (j)*WARP_SIZE;

    // Read B to regs.
    rB = dB[idn];
    // Triangular solve in regs.
    #pragma unroll
    for (int k = 0; k < 5; k++)
    {
        rA = dA[k*WARP_SIZE+idn];
        if (k%WARP_SIZE == idn)
            rB /= rA;
        double top = __shfl(rB, k%WARP_SIZE);
        if ( idn > k)
            rB -= (top*rA);
    }
    // Drop B to dev mem.
    dB[idn] = rB;
#endif
}



__device__
void dtrsv_lower_kernel_6(double *dA, double *dB )
{
#ifdef REAL
    int j = blockIdx.y * gridDim.x + blockIdx.x;
    int idn = threadIdx.x;
    double rB;
    double rA;
    dA += (j)*WARP_SIZE*WARP_SIZE;
    dB += (j)*WARP_SIZE;

    // Read B to regs.
    rB = dB[idn];
    // Triangular solve in regs.
    #pragma unroll
    for (int k = 0; k < 6; k++)
    {
        rA = dA[k*WARP_SIZE+idn];
        if (k%WARP_SIZE == idn)
            rB /= rA;
        double top = __shfl(rB, k%WARP_SIZE);
        if ( idn > k)
            rB -= (top*rA);
    }
    // Drop B to dev mem.
    dB[idn] = rB;
#endif
}



__device__
void dtrsv_lower_kernel_7(double *dA, double *dB )
{
#ifdef REAL
    int j = blockIdx.y * gridDim.x + blockIdx.x;
    int idn = threadIdx.x;
    double rB;
    double rA;
    dA += (j)*WARP_SIZE*WARP_SIZE;
    dB += (j)*WARP_SIZE;

    // Read B to regs.
    rB = dB[idn];
    // Triangular solve in regs.
    #pragma unroll
    for (int k = 0; k < 7; k++)
    {
        rA = dA[k*WARP_SIZE+idn];
        if (k%WARP_SIZE == idn)
            rB /= rA;
        double top = __shfl(rB, k%WARP_SIZE);
        if ( idn > k)
            rB -= (top*rA);
    }
    // Drop B to dev mem.
    dB[idn] = rB;
#endif
}



__device__
void dtrsv_lower_kernel_8(double *dA, double *dB )
{
#ifdef REAL
    int j = blockIdx.y * gridDim.x + blockIdx.x;
    int idn = threadIdx.x;
    double rB;
    double rA;
    dA += (j)*WARP_SIZE*WARP_SIZE;
    dB += (j)*WARP_SIZE;

    // Read B to regs.
    rB = dB[idn];
    // Triangular solve in regs.
    #pragma unroll
    for (int k = 0; k < 8; k++)
    {
        rA = dA[k*WARP_SIZE+idn];
        if (k%WARP_SIZE == idn)
            rB /= rA;
        double top = __shfl(rB, k%WARP_SIZE);
        if ( idn > k)
            rB -= (top*rA);
    }
    // Drop B to dev mem.
    dB[idn] = rB;
#endif
}



__device__
void dtrsv_lower_kernel_9(double *dA, double *dB )
{
#ifdef REAL
    int j = blockIdx.y * gridDim.x + blockIdx.x;
    int idn = threadIdx.x;
    double rB;
    double rA;
    dA += (j)*WARP_SIZE*WARP_SIZE;
    dB += (j)*WARP_SIZE;

    // Read B to regs.
    rB = dB[idn];
    // Triangular solve in regs.
    #pragma unroll
    for (int k = 0; k < 9; k++)
    {
        rA = dA[k*WARP_SIZE+idn];
        if (k%WARP_SIZE == idn)
            rB /= rA;
        double top = __shfl(rB, k%WARP_SIZE);
        if ( idn > k)
            rB -= (top*rA);
    }
    // Drop B to dev mem.
    dB[idn] = rB;
#endif
}



__device__
void dtrsv_lower_kernel_10(double *dA, double *dB )
{
#ifdef REAL
    int j = blockIdx.y * gridDim.x + blockIdx.x;
    int idn = threadIdx.x;
    double rB;
    double rA;
    dA += (j)*WARP_SIZE*WARP_SIZE;
    dB += (j)*WARP_SIZE;

    // Read B to regs.
    rB = dB[idn];
    // Triangular solve in regs.
    #pragma unroll
    for (int k = 0; k < 10; k++)
    {
        rA = dA[k*WARP_SIZE+idn];
        if (k%WARP_SIZE == idn)
            rB /= rA;
        double top = __shfl(rB, k%WARP_SIZE);
        if ( idn > k)
            rB -= (top*rA);
    }
    // Drop B to dev mem.
    dB[idn] = rB;
#endif
}



__device__
void dtrsv_lower_kernel_11(double *dA, double *dB )
{
#ifdef REAL
    int j = blockIdx.y * gridDim.x + blockIdx.x;
    int idn = threadIdx.x;
    double rB;
    double rA;
    dA += (j)*WARP_SIZE*WARP_SIZE;
    dB += (j)*WARP_SIZE;

    // Read B to regs.
    rB = dB[idn];
    // Triangular solve in regs.
    #pragma unroll
    for (int k = 0; k < 11; k++)
    {
        rA = dA[k*WARP_SIZE+idn];
        if (k%WARP_SIZE == idn)
            rB /= rA;
        double top = __shfl(rB, k%WARP_SIZE);
        if ( idn > k)
            rB -= (top*rA);
    }
    // Drop B to dev mem.
    dB[idn] = rB;
#endif
}



__device__
void dtrsv_lower_kernel_12(double *dA, double *dB )
{
#ifdef REAL
    int j = blockIdx.y * gridDim.x + blockIdx.x;
    int idn = threadIdx.x;
    double rB;
    double rA;
    dA += (j)*WARP_SIZE*WARP_SIZE;
    dB += (j)*WARP_SIZE;

    // Read B to regs.
    rB = dB[idn];
    // Triangular solve in regs.
    #pragma unroll
    for (int k = 0; k < 12; k++)
    {
        rA = dA[k*WARP_SIZE+idn];
        if (k%WARP_SIZE == idn)
            rB /= rA;
        double top = __shfl(rB, k%WARP_SIZE);
        if ( idn > k)
            rB -= (top*rA);
    }
    // Drop B to dev mem.
    dB[idn] = rB;
#endif
}



__device__
void dtrsv_lower_kernel_13(double *dA, double *dB )
{
#ifdef REAL
    int j = blockIdx.y * gridDim.x + blockIdx.x;
    int idn = threadIdx.x;
    double rB;
    double rA;
    dA += (j)*WARP_SIZE*WARP_SIZE;
    dB += (j)*WARP_SIZE;

    // Read B to regs.
    rB = dB[idn];
    // Triangular solve in regs.
    #pragma unroll
    for (int k = 0; k < 13; k++)
    {
        rA = dA[k*WARP_SIZE+idn];
        if (k%WARP_SIZE == idn)
            rB /= rA;
        double top = __shfl(rB, k%WARP_SIZE);
        if ( idn > k)
            rB -= (top*rA);
    }
    // Drop B to dev mem.
    dB[idn] = rB;
#endif
}



__device__
void dtrsv_lower_kernel_14(double *dA, double *dB )
{
#ifdef REAL
    int j = blockIdx.y * gridDim.x + blockIdx.x;
    int idn = threadIdx.x;
    double rB;
    double rA;
    dA += (j)*WARP_SIZE*WARP_SIZE;
    dB += (j)*WARP_SIZE;

    // Read B to regs.
    rB = dB[idn];
    // Triangular solve in regs.
    #pragma unroll
    for (int k = 0; k < 14; k++)
    {
        rA = dA[k*WARP_SIZE+idn];
        if (k%WARP_SIZE == idn)
            rB /= rA;
        double top = __shfl(rB, k%WARP_SIZE);
        if ( idn > k)
            rB -= (top*rA);
    }
    // Drop B to dev mem.
    dB[idn] = rB;
#endif
}



__device__
void dtrsv_lower_kernel_15(double *dA, double *dB )
{
#ifdef REAL
    int j = blockIdx.y * gridDim.x + blockIdx.x;
    int idn = threadIdx.x;
    double rB;
    double rA;
    dA += (j)*WARP_SIZE*WARP_SIZE;
    dB += (j)*WARP_SIZE;

    // Read B to regs.
    rB = dB[idn];
    // Triangular solve in regs.
    #pragma unroll
    for (int k = 0; k < 15; k++)
    {
        rA = dA[k*WARP_SIZE+idn];
        if (k%WARP_SIZE == idn)
            rB /= rA;
        double top = __shfl(rB, k%WARP_SIZE);
        if ( idn > k)
            rB -= (top*rA);
    }
    // Drop B to dev mem.
    dB[idn] = rB;
#endif
}



__device__
void dtrsv_lower_kernel_16(double *dA, double *dB )
{
#ifdef REAL
    int j = blockIdx.y * gridDim.x + blockIdx.x;
    int idn = threadIdx.x;
    double rB;
    double rA;
    dA += (j)*WARP_SIZE*WARP_SIZE;
    dB += (j)*WARP_SIZE;

    // Read B to regs.
    rB = dB[idn];
    // Triangular solve in regs.
    #pragma unroll
    for (int k = 0; k < 16; k++)
    {
        rA = dA[k*WARP_SIZE+idn];
        if (k%WARP_SIZE == idn)
            rB /= rA;
        double top = __shfl(rB, k%WARP_SIZE);
        if ( idn > k)
            rB -= (top*rA);
    }
    // Drop B to dev mem.
    dB[idn] = rB;
#endif
}



__device__
void dtrsv_lower_kernel_17(double *dA, double *dB )
{
#ifdef REAL
    int j = blockIdx.y * gridDim.x + blockIdx.x;
    int idn = threadIdx.x;
    double rB;
    double rA;
    dA += (j)*WARP_SIZE*WARP_SIZE;
    dB += (j)*WARP_SIZE;

    // Read B to regs.
    rB = dB[idn];
    // Triangular solve in regs.
    #pragma unroll
    for (int k = 0; k < 17; k++)
    {
        rA = dA[k*WARP_SIZE+idn];
        if (k%WARP_SIZE == idn)
            rB /= rA;
        double top = __shfl(rB, k%WARP_SIZE);
        if ( idn > k)
            rB -= (top*rA);
    }
    // Drop B to dev mem.
    dB[idn] = rB;
#endif
}



__device__
void dtrsv_lower_kernel_18(double *dA, double *dB )
{
#ifdef REAL
    int j = blockIdx.y * gridDim.x + blockIdx.x;
    int idn = threadIdx.x;
    double rB;
    double rA;
    dA += (j)*WARP_SIZE*WARP_SIZE;
    dB += (j)*WARP_SIZE;

    // Read B to regs.
    rB = dB[idn];
    // Triangular solve in regs.
    #pragma unroll
    for (int k = 0; k < 18; k++)
    {
        rA = dA[k*WARP_SIZE+idn];
        if (k%WARP_SIZE == idn)
            rB /= rA;
        double top = __shfl(rB, k%WARP_SIZE);
        if ( idn > k)
            rB -= (top*rA);
    }
    // Drop B to dev mem.
    dB[idn] = rB;
#endif
}



__device__
void dtrsv_lower_kernel_19(double *dA, double *dB )
{
#ifdef REAL
    int j = blockIdx.y * gridDim.x + blockIdx.x;
    int idn = threadIdx.x;
    double rB;
    double rA;
    dA += (j)*WARP_SIZE*WARP_SIZE;
    dB += (j)*WARP_SIZE;

    // Read B to regs.
    rB = dB[idn];
    // Triangular solve in regs.
    #pragma unroll
    for (int k = 0; k < 19; k++)
    {
        rA = dA[k*WARP_SIZE+idn];
        if (k%WARP_SIZE == idn)
            rB /= rA;
        double top = __shfl(rB, k%WARP_SIZE);
        if ( idn > k)
            rB -= (top*rA);
    }
    // Drop B to dev mem.
    dB[idn] = rB;
#endif
}



__device__
void dtrsv_lower_kernel_20(double *dA, double *dB )
{
#ifdef REAL
    int j = blockIdx.y * gridDim.x + blockIdx.x;
    int idn = threadIdx.x;
    double rB;
    double rA;
    dA += (j)*WARP_SIZE*WARP_SIZE;
    dB += (j)*WARP_SIZE;

    // Read B to regs.
    rB = dB[idn];
    // Triangular solve in regs.
    #pragma unroll
    for (int k = 0; k < 20; k++)
    {
        rA = dA[k*WARP_SIZE+idn];
        if (k%WARP_SIZE == idn)
            rB /= rA;
        double top = __shfl(rB, k%WARP_SIZE);
        if ( idn > k)
            rB -= (top*rA);
    }
    // Drop B to dev mem.
    dB[idn] = rB;
#endif
}



__device__
void dtrsv_lower_kernel_21(double *dA, double *dB )
{
#ifdef REAL
    int j = blockIdx.y * gridDim.x + blockIdx.x;
    int idn = threadIdx.x;
    double rB;
    double rA;
    dA += (j)*WARP_SIZE*WARP_SIZE;
    dB += (j)*WARP_SIZE;

    // Read B to regs.
    rB = dB[idn];
    // Triangular solve in regs.
    #pragma unroll
    for (int k = 0; k < 21; k++)
    {
        rA = dA[k*WARP_SIZE+idn];
        if (k%WARP_SIZE == idn)
            rB /= rA;
        double top = __shfl(rB, k%WARP_SIZE);
        if ( idn > k)
            rB -= (top*rA);
    }
    // Drop B to dev mem.
    dB[idn] = rB;
#endif
}



__device__
void dtrsv_lower_kernel_22(double *dA, double *dB )
{
#ifdef REAL
    int j = blockIdx.y * gridDim.x + blockIdx.x;
    int idn = threadIdx.x;
    double rB;
    double rA;
    dA += (j)*WARP_SIZE*WARP_SIZE;
    dB += (j)*WARP_SIZE;

    // Read B to regs.
    rB = dB[idn];
    // Triangular solve in regs.
    #pragma unroll
    for (int k = 0; k < 22; k++)
    {
        rA = dA[k*WARP_SIZE+idn];
        if (k%WARP_SIZE == idn)
            rB /= rA;
        double top = __shfl(rB, k%WARP_SIZE);
        if ( idn > k)
            rB -= (top*rA);
    }
    // Drop B to dev mem.
    dB[idn] = rB;
#endif
}



__device__
void dtrsv_lower_kernel_23(double *dA, double *dB )
{
#ifdef REAL
    int j = blockIdx.y * gridDim.x + blockIdx.x;
    int idn = threadIdx.x;
    double rB;
    double rA;
    dA += (j)*WARP_SIZE*WARP_SIZE;
    dB += (j)*WARP_SIZE;

    // Read B to regs.
    rB = dB[idn];
    // Triangular solve in regs.
    #pragma unroll
    for (int k = 0; k < 23; k++)
    {
        rA = dA[k*WARP_SIZE+idn];
        if (k%WARP_SIZE == idn)
            rB /= rA;
        double top = __shfl(rB, k%WARP_SIZE);
        if ( idn > k)
            rB -= (top*rA);
    }
    // Drop B to dev mem.
    dB[idn] = rB;
#endif
}



__device__
void dtrsv_lower_kernel_24(double *dA, double *dB )
{
#ifdef REAL
    int j = blockIdx.y * gridDim.x + blockIdx.x;
    int idn = threadIdx.x;
    double rB;
    double rA;
    dA += (j)*WARP_SIZE*WARP_SIZE;
    dB += (j)*WARP_SIZE;

    // Read B to regs.
    rB = dB[idn];
    // Triangular solve in regs.
    #pragma unroll
    for (int k = 0; k < 24; k++)
    {
        rA = dA[k*WARP_SIZE+idn];
        if (k%WARP_SIZE == idn)
            rB /= rA;
        double top = __shfl(rB, k%WARP_SIZE);
        if ( idn > k)
            rB -= (top*rA);
    }
    // Drop B to dev mem.
    dB[idn] = rB;
#endif
}



__device__
void dtrsv_lower_kernel_25(double *dA, double *dB )
{
#ifdef REAL
    int j = blockIdx.y * gridDim.x + blockIdx.x;
    int idn = threadIdx.x;
    double rB;
    double rA;
    dA += (j)*WARP_SIZE*WARP_SIZE;
    dB += (j)*WARP_SIZE;

    // Read B to regs.
    rB = dB[idn];
    // Triangular solve in regs.
    #pragma unroll
    for (int k = 0; k < 25; k++)
    {
        rA = dA[k*WARP_SIZE+idn];
        if (k%WARP_SIZE == idn)
            rB /= rA;
        double top = __shfl(rB, k%WARP_SIZE);
        if ( idn > k)
            rB -= (top*rA);
    }
    // Drop B to dev mem.
    dB[idn] = rB;
#endif
}



__device__
void dtrsv_lower_kernel_26(double *dA, double *dB )
{
#ifdef REAL
    int j = blockIdx.y * gridDim.x + blockIdx.x;
    int idn = threadIdx.x;
    double rB;
    double rA;
    dA += (j)*WARP_SIZE*WARP_SIZE;
    dB += (j)*WARP_SIZE;

    // Read B to regs.
    rB = dB[idn];
    // Triangular solve in regs.
    #pragma unroll
    for (int k = 0; k < 26; k++)
    {
        rA = dA[k*WARP_SIZE+idn];
        if (k%WARP_SIZE == idn)
            rB /= rA;
        double top = __shfl(rB, k%WARP_SIZE);
        if ( idn > k)
            rB -= (top*rA);
    }
    // Drop B to dev mem.
    dB[idn] = rB;
#endif
}



__device__
void dtrsv_lower_kernel_27(double *dA, double *dB )
{
#ifdef REAL
    int j = blockIdx.y * gridDim.x + blockIdx.x;
    int idn = threadIdx.x;
    double rB;
    double rA;
    dA += (j)*WARP_SIZE*WARP_SIZE;
    dB += (j)*WARP_SIZE;

    // Read B to regs.
    rB = dB[idn];
    // Triangular solve in regs.
    #pragma unroll
    for (int k = 0; k < 27; k++)
    {
        rA = dA[k*WARP_SIZE+idn];
        if (k%WARP_SIZE == idn)
            rB /= rA;
        double top = __shfl(rB, k%WARP_SIZE);
        if ( idn > k)
            rB -= (top*rA);
    }
    // Drop B to dev mem.
    dB[idn] = rB;
#endif
}



__device__
void dtrsv_lower_kernel_28(double *dA, double *dB )
{
#ifdef REAL
    int j = blockIdx.y * gridDim.x + blockIdx.x;
    int idn = threadIdx.x;
    double rB;
    double rA;
    dA += (j)*WARP_SIZE*WARP_SIZE;
    dB += (j)*WARP_SIZE;

    // Read B to regs.
    rB = dB[idn];
    // Triangular solve in regs.
    #pragma unroll
    for (int k = 0; k < 28; k++)
    {
        rA = dA[k*WARP_SIZE+idn];
        if (k%WARP_SIZE == idn)
            rB /= rA;
        double top = __shfl(rB, k%WARP_SIZE);
        if ( idn > k)
            rB -= (top*rA);
    }
    // Drop B to dev mem.
    dB[idn] = rB;
#endif
}



__device__
void dtrsv_lower_kernel_29(double *dA, double *dB )
{
#ifdef REAL
    int j = blockIdx.y * gridDim.x + blockIdx.x;
    int idn = threadIdx.x;
    double rB;
    double rA;
    dA += (j)*WARP_SIZE*WARP_SIZE;
    dB += (j)*WARP_SIZE;

    // Read B to regs.
    rB = dB[idn];
    // Triangular solve in regs.
    #pragma unroll
    for (int k = 0; k < 29; k++)
    {
        rA = dA[k*WARP_SIZE+idn];
        if (k%WARP_SIZE == idn)
            rB /= rA;
        double top = __shfl(rB, k%WARP_SIZE);
        if ( idn > k)
            rB -= (top*rA);
    }
    // Drop B to dev mem.
    dB[idn] = rB;
#endif
}



__device__
void dtrsv_lower_kernel_30(double *dA, double *dB )
{
#ifdef REAL
    int j = blockIdx.y * gridDim.x + blockIdx.x;
    int idn = threadIdx.x;
    double rB;
    double rA;
    dA += (j)*WARP_SIZE*WARP_SIZE;
    dB += (j)*WARP_SIZE;

    // Read B to regs.
    rB = dB[idn];
    // Triangular solve in regs.
    #pragma unroll
    for (int k = 0; k < 30; k++)
    {
        rA = dA[k*WARP_SIZE+idn];
        if (k%WARP_SIZE == idn)
            rB /= rA;
        double top = __shfl(rB, k%WARP_SIZE);
        if ( idn > k)
            rB -= (top*rA);
    }
    // Drop B to dev mem.
    dB[idn] = rB;
#endif
}



__device__
void dtrsv_lower_kernel_31(double *dA, double *dB )
{
#ifdef REAL
    int j = blockIdx.y * gridDim.x + blockIdx.x;
    int idn = threadIdx.x;
    double rB;
    double rA;
    dA += (j)*WARP_SIZE*WARP_SIZE;
    dB += (j)*WARP_SIZE;

    // Read B to regs.
    rB = dB[idn];
    // Triangular solve in regs.
    #pragma unroll
    for (int k = 0; k < 31; k++)
    {
        rA = dA[k*WARP_SIZE+idn];
        if (k%WARP_SIZE == idn)
            rB /= rA;
        double top = __shfl(rB, k%WARP_SIZE);
        if ( idn > k)
            rB -= (top*rA);
    }
    // Drop B to dev mem.
    dB[idn] = rB;
#endif
}



__device__
void dtrsv_lower_kernel_32(double *dA, double *dB )
{
#ifdef REAL
    int j = blockIdx.y * gridDim.x + blockIdx.x;
    int idn = threadIdx.x;
    double rB;
    double rA;
    dA += (j)*WARP_SIZE*WARP_SIZE;
    dB += (j)*WARP_SIZE;

    // Read B to regs.
    rB = dB[idn];
    // Triangular solve in regs.
    #pragma unroll
    for (int k = 0; k < 32; k++)
    {
        rA = dA[k*WARP_SIZE+idn];
        if (k%WARP_SIZE == idn)
            rB /= rA;
        double top = __shfl(rB, k%WARP_SIZE);
        if ( idn > k)
            rB -= (top*rA);
    }
    // Drop B to dev mem.
    dB[idn] = rB;
#endif
}


__global__
void dtrsv_lower_kernel_switch(double *dA, double *dB, int *sizes, int num_rows )
{
    int j = blockIdx.y * gridDim.x + blockIdx.x;
    if (j < num_rows) {
        int N = sizes[j];
        switch( N ) {
            case  1:
                dtrsv_lower_kernel_1( dA, dB ); break;
            case  2:
                dtrsv_lower_kernel_2( dA, dB ); break;
            case  3:
                dtrsv_lower_kernel_3( dA, dB ); break;
            case  4:
                dtrsv_lower_kernel_4( dA, dB ); break;
            case  5:
                dtrsv_lower_kernel_5( dA, dB ); break;
            case  6:
                dtrsv_lower_kernel_6( dA, dB ); break;
            case  7:
                dtrsv_lower_kernel_7( dA, dB ); break;
            case  8:
                dtrsv_lower_kernel_8( dA, dB ); break;
            case  9:
                dtrsv_lower_kernel_9( dA, dB ); break;
            case  10:
                dtrsv_lower_kernel_10( dA, dB ); break;
            case  11:
                dtrsv_lower_kernel_11( dA, dB ); break;
            case  12:
                dtrsv_lower_kernel_12( dA, dB ); break;
            case  13:
                dtrsv_lower_kernel_13( dA, dB ); break;
            case  14:
                dtrsv_lower_kernel_14( dA, dB ); break;
            case  15:
                dtrsv_lower_kernel_15( dA, dB ); break;
            case  16:
                dtrsv_lower_kernel_16( dA, dB ); break;
            case  17:
                dtrsv_lower_kernel_17( dA, dB ); break;
            case  18:
                dtrsv_lower_kernel_18( dA, dB ); break;
            case  19:
                dtrsv_lower_kernel_19( dA, dB ); break;
            case  20:
                dtrsv_lower_kernel_20( dA, dB ); break;
            case  21:
                dtrsv_lower_kernel_21( dA, dB ); break;
            case  22:
                dtrsv_lower_kernel_22( dA, dB ); break;
            case  23:
                dtrsv_lower_kernel_23( dA, dB ); break;
            case  24:
                dtrsv_lower_kernel_24( dA, dB ); break;
            case  25:
                dtrsv_lower_kernel_25( dA, dB ); break;
            case  26:
                dtrsv_lower_kernel_26( dA, dB ); break;
            case  27:
                dtrsv_lower_kernel_27( dA, dB ); break;
            case  28:
                dtrsv_lower_kernel_28( dA, dB ); break;
            case  29:
                dtrsv_lower_kernel_29( dA, dB ); break;
            case  30:
                dtrsv_lower_kernel_30( dA, dB ); break;
            case  31:
                dtrsv_lower_kernel_31( dA, dB ); break;
            case  32:
                dtrsv_lower_kernel_32( dA, dB ); break;
            default:
                dtrsv_lower_kernel_general( dA, dB, sizes ); break;
        }
    }
}
__device__
void dtrsv_upper_kernel_1(double *dA, double *dB )
{
#ifdef REAL
    int j = blockIdx.y * gridDim.x + blockIdx.x;
    int idn = threadIdx.x;
    double rB;
    double rA;
    dA += (j)*WARP_SIZE*WARP_SIZE;
    dB += (j)*WARP_SIZE;

    // Read B to regs.
    rB = dB[idn];
    // Triangular solve in regs.
    #pragma unroll
    for (int k = 1-1; k >-1; k--)
    {
        rA = dA[k*WARP_SIZE+idn];
        if (k%WARP_SIZE == idn)
            rB /= rA;
        double bottom = __shfl(rB, k%WARP_SIZE);
        if ( idn < k)
            rB -= (bottom*rA);
    }
    // Drop B to dev mem.
    dB[idn] = rB;
#endif
}



__device__
void dtrsv_upper_kernel_2(double *dA, double *dB )
{
#ifdef REAL
    int j = blockIdx.y * gridDim.x + blockIdx.x;
    int idn = threadIdx.x;
    double rB;
    double rA;
    dA += (j)*WARP_SIZE*WARP_SIZE;
    dB += (j)*WARP_SIZE;

    // Read B to regs.
    rB = dB[idn];
    // Triangular solve in regs.
    #pragma unroll
    for (int k = 2-1; k >-1; k--)
    {
        rA = dA[k*WARP_SIZE+idn];
        if (k%WARP_SIZE == idn)
            rB /= rA;
        double bottom = __shfl(rB, k%WARP_SIZE);
        if ( idn < k)
            rB -= (bottom*rA);
    }
    // Drop B to dev mem.
    dB[idn] = rB;
#endif
}



__device__
void dtrsv_upper_kernel_3(double *dA, double *dB )
{
#ifdef REAL
    int j = blockIdx.y * gridDim.x + blockIdx.x;
    int idn = threadIdx.x;
    double rB;
    double rA;
    dA += (j)*WARP_SIZE*WARP_SIZE;
    dB += (j)*WARP_SIZE;

    // Read B to regs.
    rB = dB[idn];
    // Triangular solve in regs.
    #pragma unroll
    for (int k = 3-1; k >-1; k--)
    {
        rA = dA[k*WARP_SIZE+idn];
        if (k%WARP_SIZE == idn)
            rB /= rA;
        double bottom = __shfl(rB, k%WARP_SIZE);
        if ( idn < k)
            rB -= (bottom*rA);
    }
    // Drop B to dev mem.
    dB[idn] = rB;
#endif
}



__device__
void dtrsv_upper_kernel_4(double *dA, double *dB )
{
#ifdef REAL
    int j = blockIdx.y * gridDim.x + blockIdx.x;
    int idn = threadIdx.x;
    double rB;
    double rA;
    dA += (j)*WARP_SIZE*WARP_SIZE;
    dB += (j)*WARP_SIZE;

    // Read B to regs.
    rB = dB[idn];
    // Triangular solve in regs.
    #pragma unroll
    for (int k = 4-1; k >-1; k--)
    {
        rA = dA[k*WARP_SIZE+idn];
        if (k%WARP_SIZE == idn)
            rB /= rA;
        double bottom = __shfl(rB, k%WARP_SIZE);
        if ( idn < k)
            rB -= (bottom*rA);
    }
    // Drop B to dev mem.
    dB[idn] = rB;
#endif
}



__device__
void dtrsv_upper_kernel_5(double *dA, double *dB )
{
#ifdef REAL
    int j = blockIdx.y * gridDim.x + blockIdx.x;
    int idn = threadIdx.x;
    double rB;
    double rA;
    dA += (j)*WARP_SIZE*WARP_SIZE;
    dB += (j)*WARP_SIZE;

    // Read B to regs.
    rB = dB[idn];
    // Triangular solve in regs.
    #pragma unroll
    for (int k = 5-1; k >-1; k--)
    {
        rA = dA[k*WARP_SIZE+idn];
        if (k%WARP_SIZE == idn)
            rB /= rA;
        double bottom = __shfl(rB, k%WARP_SIZE);
        if ( idn < k)
            rB -= (bottom*rA);
    }
    // Drop B to dev mem.
    dB[idn] = rB;
#endif
}



__device__
void dtrsv_upper_kernel_6(double *dA, double *dB )
{
#ifdef REAL
    int j = blockIdx.y * gridDim.x + blockIdx.x;
    int idn = threadIdx.x;
    double rB;
    double rA;
    dA += (j)*WARP_SIZE*WARP_SIZE;
    dB += (j)*WARP_SIZE;

    // Read B to regs.
    rB = dB[idn];
    // Triangular solve in regs.
    #pragma unroll
    for (int k = 6-1; k >-1; k--)
    {
        rA = dA[k*WARP_SIZE+idn];
        if (k%WARP_SIZE == idn)
            rB /= rA;
        double bottom = __shfl(rB, k%WARP_SIZE);
        if ( idn < k)
            rB -= (bottom*rA);
    }
    // Drop B to dev mem.
    dB[idn] = rB;
#endif
}



__device__
void dtrsv_upper_kernel_7(double *dA, double *dB )
{
#ifdef REAL
    int j = blockIdx.y * gridDim.x + blockIdx.x;
    int idn = threadIdx.x;
    double rB;
    double rA;
    dA += (j)*WARP_SIZE*WARP_SIZE;
    dB += (j)*WARP_SIZE;

    // Read B to regs.
    rB = dB[idn];
    // Triangular solve in regs.
    #pragma unroll
    for (int k = 7-1; k >-1; k--)
    {
        rA = dA[k*WARP_SIZE+idn];
        if (k%WARP_SIZE == idn)
            rB /= rA;
        double bottom = __shfl(rB, k%WARP_SIZE);
        if ( idn < k)
            rB -= (bottom*rA);
    }
    // Drop B to dev mem.
    dB[idn] = rB;
#endif
}



__device__
void dtrsv_upper_kernel_8(double *dA, double *dB )
{
#ifdef REAL
    int j = blockIdx.y * gridDim.x + blockIdx.x;
    int idn = threadIdx.x;
    double rB;
    double rA;
    dA += (j)*WARP_SIZE*WARP_SIZE;
    dB += (j)*WARP_SIZE;

    // Read B to regs.
    rB = dB[idn];
    // Triangular solve in regs.
    #pragma unroll
    for (int k = 8-1; k >-1; k--)
    {
        rA = dA[k*WARP_SIZE+idn];
        if (k%WARP_SIZE == idn)
            rB /= rA;
        double bottom = __shfl(rB, k%WARP_SIZE);
        if ( idn < k)
            rB -= (bottom*rA);
    }
    // Drop B to dev mem.
    dB[idn] = rB;
#endif
}



__device__
void dtrsv_upper_kernel_9(double *dA, double *dB )
{
#ifdef REAL
    int j = blockIdx.y * gridDim.x + blockIdx.x;
    int idn = threadIdx.x;
    double rB;
    double rA;
    dA += (j)*WARP_SIZE*WARP_SIZE;
    dB += (j)*WARP_SIZE;

    // Read B to regs.
    rB = dB[idn];
    // Triangular solve in regs.
    #pragma unroll
    for (int k = 9-1; k >-1; k--)
    {
        rA = dA[k*WARP_SIZE+idn];
        if (k%WARP_SIZE == idn)
            rB /= rA;
        double bottom = __shfl(rB, k%WARP_SIZE);
        if ( idn < k)
            rB -= (bottom*rA);
    }
    // Drop B to dev mem.
    dB[idn] = rB;
#endif
}



__device__
void dtrsv_upper_kernel_10(double *dA, double *dB )
{
#ifdef REAL
    int j = blockIdx.y * gridDim.x + blockIdx.x;
    int idn = threadIdx.x;
    double rB;
    double rA;
    dA += (j)*WARP_SIZE*WARP_SIZE;
    dB += (j)*WARP_SIZE;

    // Read B to regs.
    rB = dB[idn];
    // Triangular solve in regs.
    #pragma unroll
    for (int k = 10-1; k >-1; k--)
    {
        rA = dA[k*WARP_SIZE+idn];
        if (k%WARP_SIZE == idn)
            rB /= rA;
        double bottom = __shfl(rB, k%WARP_SIZE);
        if ( idn < k)
            rB -= (bottom*rA);
    }
    // Drop B to dev mem.
    dB[idn] = rB;
#endif
}



__device__
void dtrsv_upper_kernel_11(double *dA, double *dB )
{
#ifdef REAL
    int j = blockIdx.y * gridDim.x + blockIdx.x;
    int idn = threadIdx.x;
    double rB;
    double rA;
    dA += (j)*WARP_SIZE*WARP_SIZE;
    dB += (j)*WARP_SIZE;

    // Read B to regs.
    rB = dB[idn];
    // Triangular solve in regs.
    #pragma unroll
    for (int k = 11-1; k >-1; k--)
    {
        rA = dA[k*WARP_SIZE+idn];
        if (k%WARP_SIZE == idn)
            rB /= rA;
        double bottom = __shfl(rB, k%WARP_SIZE);
        if ( idn < k)
            rB -= (bottom*rA);
    }
    // Drop B to dev mem.
    dB[idn] = rB;
#endif
}



__device__
void dtrsv_upper_kernel_12(double *dA, double *dB )
{
#ifdef REAL
    int j = blockIdx.y * gridDim.x + blockIdx.x;
    int idn = threadIdx.x;
    double rB;
    double rA;
    dA += (j)*WARP_SIZE*WARP_SIZE;
    dB += (j)*WARP_SIZE;

    // Read B to regs.
    rB = dB[idn];
    // Triangular solve in regs.
    #pragma unroll
    for (int k = 12-1; k >-1; k--)
    {
        rA = dA[k*WARP_SIZE+idn];
        if (k%WARP_SIZE == idn)
            rB /= rA;
        double bottom = __shfl(rB, k%WARP_SIZE);
        if ( idn < k)
            rB -= (bottom*rA);
    }
    // Drop B to dev mem.
    dB[idn] = rB;
#endif
}



__device__
void dtrsv_upper_kernel_13(double *dA, double *dB )
{
#ifdef REAL
    int j = blockIdx.y * gridDim.x + blockIdx.x;
    int idn = threadIdx.x;
    double rB;
    double rA;
    dA += (j)*WARP_SIZE*WARP_SIZE;
    dB += (j)*WARP_SIZE;

    // Read B to regs.
    rB = dB[idn];
    // Triangular solve in regs.
    #pragma unroll
    for (int k = 13-1; k >-1; k--)
    {
        rA = dA[k*WARP_SIZE+idn];
        if (k%WARP_SIZE == idn)
            rB /= rA;
        double bottom = __shfl(rB, k%WARP_SIZE);
        if ( idn < k)
            rB -= (bottom*rA);
    }
    // Drop B to dev mem.
    dB[idn] = rB;
#endif
}



__device__
void dtrsv_upper_kernel_14(double *dA, double *dB )
{
#ifdef REAL
    int j = blockIdx.y * gridDim.x + blockIdx.x;
    int idn = threadIdx.x;
    double rB;
    double rA;
    dA += (j)*WARP_SIZE*WARP_SIZE;
    dB += (j)*WARP_SIZE;

    // Read B to regs.
    rB = dB[idn];
    // Triangular solve in regs.
    #pragma unroll
    for (int k = 14-1; k >-1; k--)
    {
        rA = dA[k*WARP_SIZE+idn];
        if (k%WARP_SIZE == idn)
            rB /= rA;
        double bottom = __shfl(rB, k%WARP_SIZE);
        if ( idn < k)
            rB -= (bottom*rA);
    }
    // Drop B to dev mem.
    dB[idn] = rB;
#endif
}



__device__
void dtrsv_upper_kernel_15(double *dA, double *dB )
{
#ifdef REAL
    int j = blockIdx.y * gridDim.x + blockIdx.x;
    int idn = threadIdx.x;
    double rB;
    double rA;
    dA += (j)*WARP_SIZE*WARP_SIZE;
    dB += (j)*WARP_SIZE;

    // Read B to regs.
    rB = dB[idn];
    // Triangular solve in regs.
    #pragma unroll
    for (int k = 15-1; k >-1; k--)
    {
        rA = dA[k*WARP_SIZE+idn];
        if (k%WARP_SIZE == idn)
            rB /= rA;
        double bottom = __shfl(rB, k%WARP_SIZE);
        if ( idn < k)
            rB -= (bottom*rA);
    }
    // Drop B to dev mem.
    dB[idn] = rB;
#endif
}



__device__
void dtrsv_upper_kernel_16(double *dA, double *dB )
{
#ifdef REAL
    int j = blockIdx.y * gridDim.x + blockIdx.x;
    int idn = threadIdx.x;
    double rB;
    double rA;
    dA += (j)*WARP_SIZE*WARP_SIZE;
    dB += (j)*WARP_SIZE;

    // Read B to regs.
    rB = dB[idn];
    // Triangular solve in regs.
    #pragma unroll
    for (int k = 16-1; k >-1; k--)
    {
        rA = dA[k*WARP_SIZE+idn];
        if (k%WARP_SIZE == idn)
            rB /= rA;
        double bottom = __shfl(rB, k%WARP_SIZE);
        if ( idn < k)
            rB -= (bottom*rA);
    }
    // Drop B to dev mem.
    dB[idn] = rB;
#endif
}



__device__
void dtrsv_upper_kernel_17(double *dA, double *dB )
{
#ifdef REAL
    int j = blockIdx.y * gridDim.x + blockIdx.x;
    int idn = threadIdx.x;
    double rB;
    double rA;
    dA += (j)*WARP_SIZE*WARP_SIZE;
    dB += (j)*WARP_SIZE;

    // Read B to regs.
    rB = dB[idn];
    // Triangular solve in regs.
    #pragma unroll
    for (int k = 17-1; k >-1; k--)
    {
        rA = dA[k*WARP_SIZE+idn];
        if (k%WARP_SIZE == idn)
            rB /= rA;
        double bottom = __shfl(rB, k%WARP_SIZE);
        if ( idn < k)
            rB -= (bottom*rA);
    }
    // Drop B to dev mem.
    dB[idn] = rB;
#endif
}



__device__
void dtrsv_upper_kernel_18(double *dA, double *dB )
{
#ifdef REAL
    int j = blockIdx.y * gridDim.x + blockIdx.x;
    int idn = threadIdx.x;
    double rB;
    double rA;
    dA += (j)*WARP_SIZE*WARP_SIZE;
    dB += (j)*WARP_SIZE;

    // Read B to regs.
    rB = dB[idn];
    // Triangular solve in regs.
    #pragma unroll
    for (int k = 18-1; k >-1; k--)
    {
        rA = dA[k*WARP_SIZE+idn];
        if (k%WARP_SIZE == idn)
            rB /= rA;
        double bottom = __shfl(rB, k%WARP_SIZE);
        if ( idn < k)
            rB -= (bottom*rA);
    }
    // Drop B to dev mem.
    dB[idn] = rB;
#endif
}



__device__
void dtrsv_upper_kernel_19(double *dA, double *dB )
{
#ifdef REAL
    int j = blockIdx.y * gridDim.x + blockIdx.x;
    int idn = threadIdx.x;
    double rB;
    double rA;
    dA += (j)*WARP_SIZE*WARP_SIZE;
    dB += (j)*WARP_SIZE;

    // Read B to regs.
    rB = dB[idn];
    // Triangular solve in regs.
    #pragma unroll
    for (int k = 19-1; k >-1; k--)
    {
        rA = dA[k*WARP_SIZE+idn];
        if (k%WARP_SIZE == idn)
            rB /= rA;
        double bottom = __shfl(rB, k%WARP_SIZE);
        if ( idn < k)
            rB -= (bottom*rA);
    }
    // Drop B to dev mem.
    dB[idn] = rB;
#endif
}



__device__
void dtrsv_upper_kernel_20(double *dA, double *dB )
{
#ifdef REAL
    int j = blockIdx.y * gridDim.x + blockIdx.x;
    int idn = threadIdx.x;
    double rB;
    double rA;
    dA += (j)*WARP_SIZE*WARP_SIZE;
    dB += (j)*WARP_SIZE;

    // Read B to regs.
    rB = dB[idn];
    // Triangular solve in regs.
    #pragma unroll
    for (int k = 20-1; k >-1; k--)
    {
        rA = dA[k*WARP_SIZE+idn];
        if (k%WARP_SIZE == idn)
            rB /= rA;
        double bottom = __shfl(rB, k%WARP_SIZE);
        if ( idn < k)
            rB -= (bottom*rA);
    }
    // Drop B to dev mem.
    dB[idn] = rB;
#endif
}



__device__
void dtrsv_upper_kernel_21(double *dA, double *dB )
{
#ifdef REAL
    int j = blockIdx.y * gridDim.x + blockIdx.x;
    int idn = threadIdx.x;
    double rB;
    double rA;
    dA += (j)*WARP_SIZE*WARP_SIZE;
    dB += (j)*WARP_SIZE;

    // Read B to regs.
    rB = dB[idn];
    // Triangular solve in regs.
    #pragma unroll
    for (int k = 21-1; k >-1; k--)
    {
        rA = dA[k*WARP_SIZE+idn];
        if (k%WARP_SIZE == idn)
            rB /= rA;
        double bottom = __shfl(rB, k%WARP_SIZE);
        if ( idn < k)
            rB -= (bottom*rA);
    }
    // Drop B to dev mem.
    dB[idn] = rB;
#endif
}



__device__
void dtrsv_upper_kernel_22(double *dA, double *dB )
{
#ifdef REAL
    int j = blockIdx.y * gridDim.x + blockIdx.x;
    int idn = threadIdx.x;
    double rB;
    double rA;
    dA += (j)*WARP_SIZE*WARP_SIZE;
    dB += (j)*WARP_SIZE;

    // Read B to regs.
    rB = dB[idn];
    // Triangular solve in regs.
    #pragma unroll
    for (int k = 22-1; k >-1; k--)
    {
        rA = dA[k*WARP_SIZE+idn];
        if (k%WARP_SIZE == idn)
            rB /= rA;
        double bottom = __shfl(rB, k%WARP_SIZE);
        if ( idn < k)
            rB -= (bottom*rA);
    }
    // Drop B to dev mem.
    dB[idn] = rB;
#endif
}



__device__
void dtrsv_upper_kernel_23(double *dA, double *dB )
{
#ifdef REAL
    int j = blockIdx.y * gridDim.x + blockIdx.x;
    int idn = threadIdx.x;
    double rB;
    double rA;
    dA += (j)*WARP_SIZE*WARP_SIZE;
    dB += (j)*WARP_SIZE;

    // Read B to regs.
    rB = dB[idn];
    // Triangular solve in regs.
    #pragma unroll
    for (int k = 23-1; k >-1; k--)
    {
        rA = dA[k*WARP_SIZE+idn];
        if (k%WARP_SIZE == idn)
            rB /= rA;
        double bottom = __shfl(rB, k%WARP_SIZE);
        if ( idn < k)
            rB -= (bottom*rA);
    }
    // Drop B to dev mem.
    dB[idn] = rB;
#endif
}



__device__
void dtrsv_upper_kernel_24(double *dA, double *dB )
{
#ifdef REAL
    int j = blockIdx.y * gridDim.x + blockIdx.x;
    int idn = threadIdx.x;
    double rB;
    double rA;
    dA += (j)*WARP_SIZE*WARP_SIZE;
    dB += (j)*WARP_SIZE;

    // Read B to regs.
    rB = dB[idn];
    // Triangular solve in regs.
    #pragma unroll
    for (int k = 24-1; k >-1; k--)
    {
        rA = dA[k*WARP_SIZE+idn];
        if (k%WARP_SIZE == idn)
            rB /= rA;
        double bottom = __shfl(rB, k%WARP_SIZE);
        if ( idn < k)
            rB -= (bottom*rA);
    }
    // Drop B to dev mem.
    dB[idn] = rB;
#endif
}



__device__
void dtrsv_upper_kernel_25(double *dA, double *dB )
{
#ifdef REAL
    int j = blockIdx.y * gridDim.x + blockIdx.x;
    int idn = threadIdx.x;
    double rB;
    double rA;
    dA += (j)*WARP_SIZE*WARP_SIZE;
    dB += (j)*WARP_SIZE;

    // Read B to regs.
    rB = dB[idn];
    // Triangular solve in regs.
    #pragma unroll
    for (int k = 25-1; k >-1; k--)
    {
        rA = dA[k*WARP_SIZE+idn];
        if (k%WARP_SIZE == idn)
            rB /= rA;
        double bottom = __shfl(rB, k%WARP_SIZE);
        if ( idn < k)
            rB -= (bottom*rA);
    }
    // Drop B to dev mem.
    dB[idn] = rB;
#endif
}



__device__
void dtrsv_upper_kernel_26(double *dA, double *dB )
{
#ifdef REAL
    int j = blockIdx.y * gridDim.x + blockIdx.x;
    int idn = threadIdx.x;
    double rB;
    double rA;
    dA += (j)*WARP_SIZE*WARP_SIZE;
    dB += (j)*WARP_SIZE;

    // Read B to regs.
    rB = dB[idn];
    // Triangular solve in regs.
    #pragma unroll
    for (int k = 26-1; k >-1; k--)
    {
        rA = dA[k*WARP_SIZE+idn];
        if (k%WARP_SIZE == idn)
            rB /= rA;
        double bottom = __shfl(rB, k%WARP_SIZE);
        if ( idn < k)
            rB -= (bottom*rA);
    }
    // Drop B to dev mem.
    dB[idn] = rB;
#endif
}



__device__
void dtrsv_upper_kernel_27(double *dA, double *dB )
{
#ifdef REAL
    int j = blockIdx.y * gridDim.x + blockIdx.x;
    int idn = threadIdx.x;
    double rB;
    double rA;
    dA += (j)*WARP_SIZE*WARP_SIZE;
    dB += (j)*WARP_SIZE;

    // Read B to regs.
    rB = dB[idn];
    // Triangular solve in regs.
    #pragma unroll
    for (int k = 27-1; k >-1; k--)
    {
        rA = dA[k*WARP_SIZE+idn];
        if (k%WARP_SIZE == idn)
            rB /= rA;
        double bottom = __shfl(rB, k%WARP_SIZE);
        if ( idn < k)
            rB -= (bottom*rA);
    }
    // Drop B to dev mem.
    dB[idn] = rB;
#endif
}



__device__
void dtrsv_upper_kernel_28(double *dA, double *dB )
{
#ifdef REAL
    int j = blockIdx.y * gridDim.x + blockIdx.x;
    int idn = threadIdx.x;
    double rB;
    double rA;
    dA += (j)*WARP_SIZE*WARP_SIZE;
    dB += (j)*WARP_SIZE;

    // Read B to regs.
    rB = dB[idn];
    // Triangular solve in regs.
    #pragma unroll
    for (int k = 28-1; k >-1; k--)
    {
        rA = dA[k*WARP_SIZE+idn];
        if (k%WARP_SIZE == idn)
            rB /= rA;
        double bottom = __shfl(rB, k%WARP_SIZE);
        if ( idn < k)
            rB -= (bottom*rA);
    }
    // Drop B to dev mem.
    dB[idn] = rB;
#endif
}



__device__
void dtrsv_upper_kernel_29(double *dA, double *dB )
{
#ifdef REAL
    int j = blockIdx.y * gridDim.x + blockIdx.x;
    int idn = threadIdx.x;
    double rB;
    double rA;
    dA += (j)*WARP_SIZE*WARP_SIZE;
    dB += (j)*WARP_SIZE;

    // Read B to regs.
    rB = dB[idn];
    // Triangular solve in regs.
    #pragma unroll
    for (int k = 29-1; k >-1; k--)
    {
        rA = dA[k*WARP_SIZE+idn];
        if (k%WARP_SIZE == idn)
            rB /= rA;
        double bottom = __shfl(rB, k%WARP_SIZE);
        if ( idn < k)
            rB -= (bottom*rA);
    }
    // Drop B to dev mem.
    dB[idn] = rB;
#endif
}



__device__
void dtrsv_upper_kernel_30(double *dA, double *dB )
{
#ifdef REAL
    int j = blockIdx.y * gridDim.x + blockIdx.x;
    int idn = threadIdx.x;
    double rB;
    double rA;
    dA += (j)*WARP_SIZE*WARP_SIZE;
    dB += (j)*WARP_SIZE;

    // Read B to regs.
    rB = dB[idn];
    // Triangular solve in regs.
    #pragma unroll
    for (int k = 30-1; k >-1; k--)
    {
        rA = dA[k*WARP_SIZE+idn];
        if (k%WARP_SIZE == idn)
            rB /= rA;
        double bottom = __shfl(rB, k%WARP_SIZE);
        if ( idn < k)
            rB -= (bottom*rA);
    }
    // Drop B to dev mem.
    dB[idn] = rB;
#endif
}



__device__
void dtrsv_upper_kernel_31(double *dA, double *dB )
{
#ifdef REAL
    int j = blockIdx.y * gridDim.x + blockIdx.x;
    int idn = threadIdx.x;
    double rB;
    double rA;
    dA += (j)*WARP_SIZE*WARP_SIZE;
    dB += (j)*WARP_SIZE;

    // Read B to regs.
    rB = dB[idn];
    // Triangular solve in regs.
    #pragma unroll
    for (int k = 31-1; k >-1; k--)
    {
        rA = dA[k*WARP_SIZE+idn];
        if (k%WARP_SIZE == idn)
            rB /= rA;
        double bottom = __shfl(rB, k%WARP_SIZE);
        if ( idn < k)
            rB -= (bottom*rA);
    }
    // Drop B to dev mem.
    dB[idn] = rB;
#endif
}



__device__
void dtrsv_upper_kernel_32(double *dA, double *dB )
{
#ifdef REAL
    int j = blockIdx.y * gridDim.x + blockIdx.x;
    int idn = threadIdx.x;
    double rB;
    double rA;
    dA += (j)*WARP_SIZE*WARP_SIZE;
    dB += (j)*WARP_SIZE;

    // Read B to regs.
    rB = dB[idn];
    // Triangular solve in regs.
    #pragma unroll
    for (int k = 32-1; k >-1; k--)
    {
        rA = dA[k*WARP_SIZE+idn];
        if (k%WARP_SIZE == idn)
            rB /= rA;
        double bottom = __shfl(rB, k%WARP_SIZE);
        if ( idn < k)
            rB -= (bottom*rA);
    }
    // Drop B to dev mem.
    dB[idn] = rB;
#endif
}


__global__
void dtrsv_upper_kernel_switch(double *dA, double *dB, int *sizes, int num_rows )
{
    int j = blockIdx.y * gridDim.x + blockIdx.x;
    if (j < num_rows) {
        int N = sizes[j];
        switch( N ) {
            case  1:
                dtrsv_upper_kernel_1( dA, dB ); break;
            case  2:
                dtrsv_upper_kernel_2( dA, dB ); break;
            case  3:
                dtrsv_upper_kernel_3( dA, dB ); break;
            case  4:
                dtrsv_upper_kernel_4( dA, dB ); break;
            case  5:
                dtrsv_upper_kernel_5( dA, dB ); break;
            case  6:
                dtrsv_upper_kernel_6( dA, dB ); break;
            case  7:
                dtrsv_upper_kernel_7( dA, dB ); break;
            case  8:
                dtrsv_upper_kernel_8( dA, dB ); break;
            case  9:
                dtrsv_upper_kernel_9( dA, dB ); break;
            case  10:
                dtrsv_upper_kernel_10( dA, dB ); break;
            case  11:
                dtrsv_upper_kernel_11( dA, dB ); break;
            case  12:
                dtrsv_upper_kernel_12( dA, dB ); break;
            case  13:
                dtrsv_upper_kernel_13( dA, dB ); break;
            case  14:
                dtrsv_upper_kernel_14( dA, dB ); break;
            case  15:
                dtrsv_upper_kernel_15( dA, dB ); break;
            case  16:
                dtrsv_upper_kernel_16( dA, dB ); break;
            case  17:
                dtrsv_upper_kernel_17( dA, dB ); break;
            case  18:
                dtrsv_upper_kernel_18( dA, dB ); break;
            case  19:
                dtrsv_upper_kernel_19( dA, dB ); break;
            case  20:
                dtrsv_upper_kernel_20( dA, dB ); break;
            case  21:
                dtrsv_upper_kernel_21( dA, dB ); break;
            case  22:
                dtrsv_upper_kernel_22( dA, dB ); break;
            case  23:
                dtrsv_upper_kernel_23( dA, dB ); break;
            case  24:
                dtrsv_upper_kernel_24( dA, dB ); break;
            case  25:
                dtrsv_upper_kernel_25( dA, dB ); break;
            case  26:
                dtrsv_upper_kernel_26( dA, dB ); break;
            case  27:
                dtrsv_upper_kernel_27( dA, dB ); break;
            case  28:
                dtrsv_upper_kernel_28( dA, dB ); break;
            case  29:
                dtrsv_upper_kernel_29( dA, dB ); break;
            case  30:
                dtrsv_upper_kernel_30( dA, dB ); break;
            case  31:
                dtrsv_upper_kernel_31( dA, dB ); break;
            case  32:
                dtrsv_upper_kernel_32( dA, dB ); break;
            default:
                dtrsv_upper_kernel_general( dA, dB, sizes ); break;
        }
    }
}
#endif
#endif
/**
    Purpose
    -------
    Does all triangular solves

    Arguments
    ---------


    @param[in]
    uplotype    magma_uplo_t
                lower or upper triangular

    @param[in]
    transtype   magma_trans_t
                possibility for transposed matrix

    @param[in]
    diagtype    magma_diag_t
                unit diagonal or not

    @param[in]
    L           magma_d_matrix
                Matrix in CSR format

    @param[in]
    LC          magma_d_matrix
                same matrix, also CSR, but col-major

    @param[out]
    sizes       magma_int_t*
                Number of Elements that are replaced.

    @param[out]
    locations   magma_int_t*
                Array indicating the locations.

    @param[out]
    trisystems  double*
                trisystems

    @param[out]
    rhs         double*
                right-hand sides

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_daux
    ********************************************************************/

extern "C" magma_int_t
magma_dmtrisolve_batched_gpu(
    magma_uplo_t uplotype,
    magma_trans_t transtype,
    magma_diag_t diagtype,
    magma_d_matrix L,
    magma_d_matrix LC,
    magma_index_t *sizes,
    magma_index_t *locations,
    double *trisystems,
    double *rhs,
    magma_queue_t queue )
{
    magma_int_t info = 0;

    int blocksize1 = WARP_SIZE;
    int blocksize2 = 1;
    int dimgrid1 = min( int( sqrt( double( LC.num_rows ))), 65535 );
    int dimgrid2 = min(magma_ceildiv( LC.num_rows, dimgrid1 ), 65535);
    int dimgrid3 = magma_ceildiv( LC.num_rows, dimgrid1*dimgrid2 );

    dim3 grid( dimgrid1, dimgrid2, dimgrid3 );
    dim3 block( blocksize1, blocksize2, 1 );

#if (CUDA_VERSION >= 7000)
#if (CUDA_ARCH >= 300)
    if( uplotype == MagmaLower ){
        //cudaProfilerStart();
        dtrsv_lower_kernel_switch<<< grid, block, 0, queue->cuda_stream() >>>(
                trisystems,
                rhs,
                sizes,
                LC.num_rows );
        //cudaProfilerStop();
    } else {
        dtrsv_upper_kernel_switch<<< grid, block, 0, queue->cuda_stream() >>>(
                trisystems,
                rhs,
                sizes,
                LC.num_rows );
    }
#endif
#endif

    return info;
}
