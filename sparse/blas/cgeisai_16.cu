/*
    -- MAGMA (version 2.2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date November 2016

       @generated from sparse/blas/zgeisai_16.cu, normal z -> c, Sun Nov 20 20:20:43 2016

*/
#include "magmasparse_internal.h"

#define PRECISION_c
#define COMPLEX
#define BLOCKSIZE 16
#define WARP_SIZE 16
#define WRP 16
#define WRQ 4


#include <cuda.h>  // for CUDA_VERSION

#if (CUDA_VERSION >= 7000)

__device__
void ctrsv_lower_16kernel_general(magmaFloatComplex *dA, magmaFloatComplex *dB, int *sizes)
{
#if (defined( REAL ) && ( __CUDA_ARCH__ >= 300 ))
    int j = blockIdx.y * gridDim.x + blockIdx.x;
    int idn = threadIdx.x;

    magmaFloatComplex rB[ 2 ];
    magmaFloatComplex rA[ 2 ];

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

        magmaFloatComplex top = __shfl(rB[k/WARP_SIZE], k%WARP_SIZE);

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
void ctrsv_upper_16kernel_general(magmaFloatComplex *dA, magmaFloatComplex *dB, int *sizes)
{
#if (defined( REAL ) && ( __CUDA_ARCH__ >= 300 ))
    int j = blockIdx.y * gridDim.x + blockIdx.x;
    int idn = threadIdx.x;

    magmaFloatComplex rB[ 2 ];
    magmaFloatComplex rA[ 2 ];

    int n;
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

        magmaFloatComplex top = __shfl(rB[k/WARP_SIZE], k%WARP_SIZE);

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
void ctrsv_lower_16kernel_1(magmaFloatComplex *dA, magmaFloatComplex *dB )
{
#if (defined( REAL ) && ( __CUDA_ARCH__ >= 300 ))
    int j = blockIdx.y * gridDim.x + blockIdx.x;
    int idn = threadIdx.x;
    magmaFloatComplex rB;
    magmaFloatComplex rA;
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
        magmaFloatComplex top = __shfl(rB, k%WARP_SIZE);
        if ( idn > k)
            rB -= (top*rA);
    }
    // Drop B to dev mem.
    dB[idn] = rB;
#endif
}



__device__
void ctrsv_lower_16kernel_2(magmaFloatComplex *dA, magmaFloatComplex *dB )
{
#if (defined( REAL ) && ( __CUDA_ARCH__ >= 300 ))
    int j = blockIdx.y * gridDim.x + blockIdx.x;
    int idn = threadIdx.x;
    magmaFloatComplex rB;
    magmaFloatComplex rA;
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
        magmaFloatComplex top = __shfl(rB, k%WARP_SIZE);
        if ( idn > k)
            rB -= (top*rA);
    }
    // Drop B to dev mem.
    dB[idn] = rB;
#endif
}



__device__
void ctrsv_lower_16kernel_3(magmaFloatComplex *dA, magmaFloatComplex *dB )
{
#if (defined( REAL ) && ( __CUDA_ARCH__ >= 300 ))
    int j = blockIdx.y * gridDim.x + blockIdx.x;
    int idn = threadIdx.x;
    magmaFloatComplex rB;
    magmaFloatComplex rA;
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
        magmaFloatComplex top = __shfl(rB, k%WARP_SIZE);
        if ( idn > k)
            rB -= (top*rA);
    }
    // Drop B to dev mem.
    dB[idn] = rB;
#endif
}



__device__
void ctrsv_lower_16kernel_4(magmaFloatComplex *dA, magmaFloatComplex *dB )
{
#if (defined( REAL ) && ( __CUDA_ARCH__ >= 300 ))
    int j = blockIdx.y * gridDim.x + blockIdx.x;
    int idn = threadIdx.x;
    magmaFloatComplex rB;
    magmaFloatComplex rA;
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
        magmaFloatComplex top = __shfl(rB, k%WARP_SIZE);
        if ( idn > k)
            rB -= (top*rA);
    }
    // Drop B to dev mem.
    dB[idn] = rB;
#endif
}



__device__
void ctrsv_lower_16kernel_5(magmaFloatComplex *dA, magmaFloatComplex *dB )
{
#if (defined( REAL ) && ( __CUDA_ARCH__ >= 300 ))
    int j = blockIdx.y * gridDim.x + blockIdx.x;
    int idn = threadIdx.x;
    magmaFloatComplex rB;
    magmaFloatComplex rA;
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
        magmaFloatComplex top = __shfl(rB, k%WARP_SIZE);
        if ( idn > k)
            rB -= (top*rA);
    }
    // Drop B to dev mem.
    dB[idn] = rB;
#endif
}



__device__
void ctrsv_lower_16kernel_6(magmaFloatComplex *dA, magmaFloatComplex *dB )
{
#if (defined( REAL ) && ( __CUDA_ARCH__ >= 300 ))
    int j = blockIdx.y * gridDim.x + blockIdx.x;
    int idn = threadIdx.x;
    magmaFloatComplex rB;
    magmaFloatComplex rA;
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
        magmaFloatComplex top = __shfl(rB, k%WARP_SIZE);
        if ( idn > k)
            rB -= (top*rA);
    }
    // Drop B to dev mem.
    dB[idn] = rB;
#endif
}



__device__
void ctrsv_lower_16kernel_7(magmaFloatComplex *dA, magmaFloatComplex *dB )
{
#if (defined( REAL ) && ( __CUDA_ARCH__ >= 300 ))
    int j = blockIdx.y * gridDim.x + blockIdx.x;
    int idn = threadIdx.x;
    magmaFloatComplex rB;
    magmaFloatComplex rA;
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
        magmaFloatComplex top = __shfl(rB, k%WARP_SIZE);
        if ( idn > k)
            rB -= (top*rA);
    }
    // Drop B to dev mem.
    dB[idn] = rB;
#endif
}



__device__
void ctrsv_lower_16kernel_8(magmaFloatComplex *dA, magmaFloatComplex *dB )
{
#if (defined( REAL ) && ( __CUDA_ARCH__ >= 300 ))
    int j = blockIdx.y * gridDim.x + blockIdx.x;
    int idn = threadIdx.x;
    magmaFloatComplex rB;
    magmaFloatComplex rA;
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
        magmaFloatComplex top = __shfl(rB, k%WARP_SIZE);
        if ( idn > k)
            rB -= (top*rA);
    }
    // Drop B to dev mem.
    dB[idn] = rB;
#endif
}



__device__
void ctrsv_lower_16kernel_9(magmaFloatComplex *dA, magmaFloatComplex *dB )
{
#if (defined( REAL ) && ( __CUDA_ARCH__ >= 300 ))
    int j = blockIdx.y * gridDim.x + blockIdx.x;
    int idn = threadIdx.x;
    magmaFloatComplex rB;
    magmaFloatComplex rA;
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
        magmaFloatComplex top = __shfl(rB, k%WARP_SIZE);
        if ( idn > k)
            rB -= (top*rA);
    }
    // Drop B to dev mem.
    dB[idn] = rB;
#endif
}



__device__
void ctrsv_lower_16kernel_10(magmaFloatComplex *dA, magmaFloatComplex *dB )
{
#if (defined( REAL ) && ( __CUDA_ARCH__ >= 300 ))
    int j = blockIdx.y * gridDim.x + blockIdx.x;
    int idn = threadIdx.x;
    magmaFloatComplex rB;
    magmaFloatComplex rA;
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
        magmaFloatComplex top = __shfl(rB, k%WARP_SIZE);
        if ( idn > k)
            rB -= (top*rA);
    }
    // Drop B to dev mem.
    dB[idn] = rB;
#endif
}



__device__
void ctrsv_lower_16kernel_11(magmaFloatComplex *dA, magmaFloatComplex *dB )
{
#if (defined( REAL ) && ( __CUDA_ARCH__ >= 300 ))
    int j = blockIdx.y * gridDim.x + blockIdx.x;
    int idn = threadIdx.x;
    magmaFloatComplex rB;
    magmaFloatComplex rA;
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
        magmaFloatComplex top = __shfl(rB, k%WARP_SIZE);
        if ( idn > k)
            rB -= (top*rA);
    }
    // Drop B to dev mem.
    dB[idn] = rB;
#endif
}



__device__
void ctrsv_lower_16kernel_12(magmaFloatComplex *dA, magmaFloatComplex *dB )
{
#if (defined( REAL ) && ( __CUDA_ARCH__ >= 300 ))
    int j = blockIdx.y * gridDim.x + blockIdx.x;
    int idn = threadIdx.x;
    magmaFloatComplex rB;
    magmaFloatComplex rA;
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
        magmaFloatComplex top = __shfl(rB, k%WARP_SIZE);
        if ( idn > k)
            rB -= (top*rA);
    }
    // Drop B to dev mem.
    dB[idn] = rB;
#endif
}



__device__
void ctrsv_lower_16kernel_13(magmaFloatComplex *dA, magmaFloatComplex *dB )
{
#if (defined( REAL ) && ( __CUDA_ARCH__ >= 300 ))
    int j = blockIdx.y * gridDim.x + blockIdx.x;
    int idn = threadIdx.x;
    magmaFloatComplex rB;
    magmaFloatComplex rA;
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
        magmaFloatComplex top = __shfl(rB, k%WARP_SIZE);
        if ( idn > k)
            rB -= (top*rA);
    }
    // Drop B to dev mem.
    dB[idn] = rB;
#endif
}



__device__
void ctrsv_lower_16kernel_14(magmaFloatComplex *dA, magmaFloatComplex *dB )
{
#if (defined( REAL ) && ( __CUDA_ARCH__ >= 300 ))
    int j = blockIdx.y * gridDim.x + blockIdx.x;
    int idn = threadIdx.x;
    magmaFloatComplex rB;
    magmaFloatComplex rA;
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
        magmaFloatComplex top = __shfl(rB, k%WARP_SIZE);
        if ( idn > k)
            rB -= (top*rA);
    }
    // Drop B to dev mem.
    dB[idn] = rB;
#endif
}



__device__
void ctrsv_lower_16kernel_15(magmaFloatComplex *dA, magmaFloatComplex *dB )
{
#if (defined( REAL ) && ( __CUDA_ARCH__ >= 300 ))
    int j = blockIdx.y * gridDim.x + blockIdx.x;
    int idn = threadIdx.x;
    magmaFloatComplex rB;
    magmaFloatComplex rA;
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
        magmaFloatComplex top = __shfl(rB, k%WARP_SIZE);
        if ( idn > k)
            rB -= (top*rA);
    }
    // Drop B to dev mem.
    dB[idn] = rB;
#endif
}



__device__
void ctrsv_lower_16kernel_16(magmaFloatComplex *dA, magmaFloatComplex *dB )
{
#if (defined( REAL ) && ( __CUDA_ARCH__ >= 300 ))
    int j = blockIdx.y * gridDim.x + blockIdx.x;
    int idn = threadIdx.x;
    magmaFloatComplex rB;
    magmaFloatComplex rA;
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
        magmaFloatComplex top = __shfl(rB, k%WARP_SIZE);
        if ( idn > k)
            rB -= (top*rA);
    }
    // Drop B to dev mem.
    dB[idn] = rB;
#endif
}


__global__
void ctrsv_lower_16kernel_switch(magmaFloatComplex *dA, magmaFloatComplex *dB, int *sizes, int num_rows )
{
    int j = blockIdx.y * gridDim.x + blockIdx.x;
    if (j < num_rows) {
        int N = sizes[j];
        switch( N ) {
            case  1:
                ctrsv_lower_16kernel_1( dA, dB ); break;
            case  2:
                ctrsv_lower_16kernel_2( dA, dB ); break;
            case  3:
                ctrsv_lower_16kernel_3( dA, dB ); break;
            case  4:
                ctrsv_lower_16kernel_4( dA, dB ); break;
            case  5:
                ctrsv_lower_16kernel_5( dA, dB ); break;
            case  6:
                ctrsv_lower_16kernel_6( dA, dB ); break;
            case  7:
                ctrsv_lower_16kernel_7( dA, dB ); break;
            case  8:
                ctrsv_lower_16kernel_8( dA, dB ); break;
            case  9:
                ctrsv_lower_16kernel_9( dA, dB ); break;
            case  10:
                ctrsv_lower_16kernel_10( dA, dB ); break;
            case  11:
                ctrsv_lower_16kernel_11( dA, dB ); break;
            case  12:
                ctrsv_lower_16kernel_12( dA, dB ); break;
            case  13:
                ctrsv_lower_16kernel_13( dA, dB ); break;
            case  14:
                ctrsv_lower_16kernel_14( dA, dB ); break;
            case  15:
                ctrsv_lower_16kernel_15( dA, dB ); break;
            case  16:
                ctrsv_lower_16kernel_16( dA, dB ); break;
            default:
                ctrsv_lower_16kernel_general( dA, dB, sizes ); break;
        }
    }
}
__device__
void ctrsv_upper_16kernel_1(magmaFloatComplex *dA, magmaFloatComplex *dB )
{
#if (defined( REAL ) && ( __CUDA_ARCH__ >= 300 ))
    int j = blockIdx.y * gridDim.x + blockIdx.x;
    int idn = threadIdx.x;
    magmaFloatComplex rB;
    magmaFloatComplex rA;
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
        magmaFloatComplex bottom = __shfl(rB, k%WARP_SIZE);
        if ( idn < k)
            rB -= (bottom*rA);
    }
    // Drop B to dev mem.
    dB[idn] = rB;
#endif
}



__device__
void ctrsv_upper_16kernel_2(magmaFloatComplex *dA, magmaFloatComplex *dB )
{
#if (defined( REAL ) && ( __CUDA_ARCH__ >= 300 ))
    int j = blockIdx.y * gridDim.x + blockIdx.x;
    int idn = threadIdx.x;
    magmaFloatComplex rB;
    magmaFloatComplex rA;
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
        magmaFloatComplex bottom = __shfl(rB, k%WARP_SIZE);
        if ( idn < k)
            rB -= (bottom*rA);
    }
    // Drop B to dev mem.
    dB[idn] = rB;
#endif
}



__device__
void ctrsv_upper_16kernel_3(magmaFloatComplex *dA, magmaFloatComplex *dB )
{
#if (defined( REAL ) && ( __CUDA_ARCH__ >= 300 ))
    int j = blockIdx.y * gridDim.x + blockIdx.x;
    int idn = threadIdx.x;
    magmaFloatComplex rB;
    magmaFloatComplex rA;
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
        magmaFloatComplex bottom = __shfl(rB, k%WARP_SIZE);
        if ( idn < k)
            rB -= (bottom*rA);
    }
    // Drop B to dev mem.
    dB[idn] = rB;
#endif
}



__device__
void ctrsv_upper_16kernel_4(magmaFloatComplex *dA, magmaFloatComplex *dB )
{
#if (defined( REAL ) && ( __CUDA_ARCH__ >= 300 ))
    int j = blockIdx.y * gridDim.x + blockIdx.x;
    int idn = threadIdx.x;
    magmaFloatComplex rB;
    magmaFloatComplex rA;
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
        magmaFloatComplex bottom = __shfl(rB, k%WARP_SIZE);
        if ( idn < k)
            rB -= (bottom*rA);
    }
    // Drop B to dev mem.
    dB[idn] = rB;
#endif
}



__device__
void ctrsv_upper_16kernel_5(magmaFloatComplex *dA, magmaFloatComplex *dB )
{
#if (defined( REAL ) && ( __CUDA_ARCH__ >= 300 ))
    int j = blockIdx.y * gridDim.x + blockIdx.x;
    int idn = threadIdx.x;
    magmaFloatComplex rB;
    magmaFloatComplex rA;
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
        magmaFloatComplex bottom = __shfl(rB, k%WARP_SIZE);
        if ( idn < k)
            rB -= (bottom*rA);
    }
    // Drop B to dev mem.
    dB[idn] = rB;
#endif
}



__device__
void ctrsv_upper_16kernel_6(magmaFloatComplex *dA, magmaFloatComplex *dB )
{
#if (defined( REAL ) && ( __CUDA_ARCH__ >= 300 ))
    int j = blockIdx.y * gridDim.x + blockIdx.x;
    int idn = threadIdx.x;
    magmaFloatComplex rB;
    magmaFloatComplex rA;
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
        magmaFloatComplex bottom = __shfl(rB, k%WARP_SIZE);
        if ( idn < k)
            rB -= (bottom*rA);
    }
    // Drop B to dev mem.
    dB[idn] = rB;
#endif
}



__device__
void ctrsv_upper_16kernel_7(magmaFloatComplex *dA, magmaFloatComplex *dB )
{
#if (defined( REAL ) && ( __CUDA_ARCH__ >= 300 ))
    int j = blockIdx.y * gridDim.x + blockIdx.x;
    int idn = threadIdx.x;
    magmaFloatComplex rB;
    magmaFloatComplex rA;
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
        magmaFloatComplex bottom = __shfl(rB, k%WARP_SIZE);
        if ( idn < k)
            rB -= (bottom*rA);
    }
    // Drop B to dev mem.
    dB[idn] = rB;
#endif
}



__device__
void ctrsv_upper_16kernel_8(magmaFloatComplex *dA, magmaFloatComplex *dB )
{
#if (defined( REAL ) && ( __CUDA_ARCH__ >= 300 ))
    int j = blockIdx.y * gridDim.x + blockIdx.x;
    int idn = threadIdx.x;
    magmaFloatComplex rB;
    magmaFloatComplex rA;
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
        magmaFloatComplex bottom = __shfl(rB, k%WARP_SIZE);
        if ( idn < k)
            rB -= (bottom*rA);
    }
    // Drop B to dev mem.
    dB[idn] = rB;
#endif
}



__device__
void ctrsv_upper_16kernel_9(magmaFloatComplex *dA, magmaFloatComplex *dB )
{
#if (defined( REAL ) && ( __CUDA_ARCH__ >= 300 ))
    int j = blockIdx.y * gridDim.x + blockIdx.x;
    int idn = threadIdx.x;
    magmaFloatComplex rB;
    magmaFloatComplex rA;
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
        magmaFloatComplex bottom = __shfl(rB, k%WARP_SIZE);
        if ( idn < k)
            rB -= (bottom*rA);
    }
    // Drop B to dev mem.
    dB[idn] = rB;
#endif
}



__device__
void ctrsv_upper_16kernel_10(magmaFloatComplex *dA, magmaFloatComplex *dB )
{
#if (defined( REAL ) && ( __CUDA_ARCH__ >= 300 ))
    int j = blockIdx.y * gridDim.x + blockIdx.x;
    int idn = threadIdx.x;
    magmaFloatComplex rB;
    magmaFloatComplex rA;
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
        magmaFloatComplex bottom = __shfl(rB, k%WARP_SIZE);
        if ( idn < k)
            rB -= (bottom*rA);
    }
    // Drop B to dev mem.
    dB[idn] = rB;
#endif
}



__device__
void ctrsv_upper_16kernel_11(magmaFloatComplex *dA, magmaFloatComplex *dB )
{
#if (defined( REAL ) && ( __CUDA_ARCH__ >= 300 ))
    int j = blockIdx.y * gridDim.x + blockIdx.x;
    int idn = threadIdx.x;
    magmaFloatComplex rB;
    magmaFloatComplex rA;
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
        magmaFloatComplex bottom = __shfl(rB, k%WARP_SIZE);
        if ( idn < k)
            rB -= (bottom*rA);
    }
    // Drop B to dev mem.
    dB[idn] = rB;
#endif
}



__device__
void ctrsv_upper_16kernel_12(magmaFloatComplex *dA, magmaFloatComplex *dB )
{
#if (defined( REAL ) && ( __CUDA_ARCH__ >= 300 ))
    int j = blockIdx.y * gridDim.x + blockIdx.x;
    int idn = threadIdx.x;
    magmaFloatComplex rB;
    magmaFloatComplex rA;
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
        magmaFloatComplex bottom = __shfl(rB, k%WARP_SIZE);
        if ( idn < k)
            rB -= (bottom*rA);
    }
    // Drop B to dev mem.
    dB[idn] = rB;
#endif
}



__device__
void ctrsv_upper_16kernel_13(magmaFloatComplex *dA, magmaFloatComplex *dB )
{
#if (defined( REAL ) && ( __CUDA_ARCH__ >= 300 ))
    int j = blockIdx.y * gridDim.x + blockIdx.x;
    int idn = threadIdx.x;
    magmaFloatComplex rB;
    magmaFloatComplex rA;
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
        magmaFloatComplex bottom = __shfl(rB, k%WARP_SIZE);
        if ( idn < k)
            rB -= (bottom*rA);
    }
    // Drop B to dev mem.
    dB[idn] = rB;
#endif
}



__device__
void ctrsv_upper_16kernel_14(magmaFloatComplex *dA, magmaFloatComplex *dB )
{
#if (defined( REAL ) && ( __CUDA_ARCH__ >= 300 ))
    int j = blockIdx.y * gridDim.x + blockIdx.x;
    int idn = threadIdx.x;
    magmaFloatComplex rB;
    magmaFloatComplex rA;
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
        magmaFloatComplex bottom = __shfl(rB, k%WARP_SIZE);
        if ( idn < k)
            rB -= (bottom*rA);
    }
    // Drop B to dev mem.
    dB[idn] = rB;
#endif
}



__device__
void ctrsv_upper_16kernel_15(magmaFloatComplex *dA, magmaFloatComplex *dB )
{
#if (defined( REAL ) && ( __CUDA_ARCH__ >= 300 ))
    int j = blockIdx.y * gridDim.x + blockIdx.x;
    int idn = threadIdx.x;
    magmaFloatComplex rB;
    magmaFloatComplex rA;
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
        magmaFloatComplex bottom = __shfl(rB, k%WARP_SIZE);
        if ( idn < k)
            rB -= (bottom*rA);
    }
    // Drop B to dev mem.
    dB[idn] = rB;
#endif
}



__device__
void ctrsv_upper_16kernel_16(magmaFloatComplex *dA, magmaFloatComplex *dB )
{
#if (defined( REAL ) && ( __CUDA_ARCH__ >= 300 ))
    int j = blockIdx.y * gridDim.x + blockIdx.x;
    int idn = threadIdx.x;
    magmaFloatComplex rB;
    magmaFloatComplex rA;
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
        magmaFloatComplex bottom = __shfl(rB, k%WARP_SIZE);
        if ( idn < k)
            rB -= (bottom*rA);
    }
    // Drop B to dev mem.
    dB[idn] = rB;
#endif
}


__global__
void ctrsv_upper_16kernel_switch(magmaFloatComplex *dA, magmaFloatComplex *dB, int *sizes, int num_rows )
{
    int j = blockIdx.y * gridDim.x + blockIdx.x;
    if (j < num_rows) {
        int N = sizes[j];
        switch( N ) {
            case  1:
                ctrsv_upper_16kernel_1( dA, dB ); break;
            case  2:
                ctrsv_upper_16kernel_2( dA, dB ); break;
            case  3:
                ctrsv_upper_16kernel_3( dA, dB ); break;
            case  4:
                ctrsv_upper_16kernel_4( dA, dB ); break;
            case  5:
                ctrsv_upper_16kernel_5( dA, dB ); break;
            case  6:
                ctrsv_upper_16kernel_6( dA, dB ); break;
            case  7:
                ctrsv_upper_16kernel_7( dA, dB ); break;
            case  8:
                ctrsv_upper_16kernel_8( dA, dB ); break;
            case  9:
                ctrsv_upper_16kernel_9( dA, dB ); break;
            case  10:
                ctrsv_upper_16kernel_10( dA, dB ); break;
            case  11:
                ctrsv_upper_16kernel_11( dA, dB ); break;
            case  12:
                ctrsv_upper_16kernel_12( dA, dB ); break;
            case  13:
                ctrsv_upper_16kernel_13( dA, dB ); break;
            case  14:
                ctrsv_upper_16kernel_14( dA, dB ); break;
            case  15:
                ctrsv_upper_16kernel_15( dA, dB ); break;
            case  16:
                ctrsv_upper_16kernel_16( dA, dB ); break;
            default:
                ctrsv_upper_16kernel_general( dA, dB, sizes ); break;
        }
    }
}


// initialize arrays with zero
__global__ void
magma_cgpumemzero_16kernel(
    magmaFloatComplex * d,
    int n,
    int dim_x,
    int dim_y )
{
    int i = blockIdx.y * gridDim.x + blockIdx.x;
    int idx = threadIdx.x;

    if( i >= n ){
       return;
    }
    if( idx >= dim_x ){
       return;
    }

    for( int j=0; j<dim_y; j++)
        d[ i*dim_x*dim_y + j*dim_y + idx ] = MAGMA_C_MAKE( 0.0, 0.0 );
}

__global__ void
magma_clocations_lower_16kernel(
    magma_int_t n,
    magma_index_t *row,
    magma_index_t *col,
    magmaFloatComplex *val,
    magma_index_t *sizes,
    magma_index_t *locations,
    magmaFloatComplex *trisystems,
    magmaFloatComplex *rhs )
{
    int i = threadIdx.x;
    int j = blockIdx.y * gridDim.x + blockIdx.x;

    if( j >= n ){
        return;
    }
    int start = row[j];
    int end = row[j+1];
    int count = end-start;
    if( i == 0 ){
        sizes[j] = count;
        rhs[ j*WARP_SIZE ] = MAGMA_C_ONE;
    }

    if ( i<count ){
        locations[ j*WARP_SIZE + i ] = col[ row[j]+i ];
    }
}// kernel


__global__ void
magma_clocations_trunc_lower_16kernel(
    magma_int_t n,
    magma_index_t *row,
    magma_index_t *col,
    magmaFloatComplex *val,
    magma_index_t *sizes,
    magma_index_t *locations,
    magmaFloatComplex *trisystems,
    magmaFloatComplex *rhs )
{
    int i = threadIdx.x;
    int j = blockIdx.y * gridDim.x + blockIdx.x;

    if( j >= n ){
        return;
    }
    int start = row[j];
    int end = row[j+1];
    int count = end-start;

    // normal case
    if( count <= BLOCKSIZE ){ // normal case
        if( i == 0 ){
            sizes[j] = count;
            rhs[ j*WARP_SIZE ] = MAGMA_C_ONE;
        }
        if ( i<count ){
            locations[ j*WARP_SIZE + i ] = col[ row[j]+i ];
        }
    }
    else {
        // truncate in this row to the blocksize,
        // take only the 16 elements close to the main diagonal into account
        count = BLOCKSIZE;
        if (i == 0) {
            sizes[j] = count;
            rhs[ j*WARP_SIZE ] = MAGMA_C_ONE;
        }

        locations[ j*WARP_SIZE + i ] = col[ row[j+1]-BLOCKSIZE+i ];
    }
}// kernel



__global__ void
magma_clocations_upper_16kernel(
    magma_int_t n,
    magma_index_t *row,
    magma_index_t *col,
    magmaFloatComplex *val,
    magma_index_t *sizes,
    magma_index_t *locations,
    magmaFloatComplex *trisystems,
    magmaFloatComplex *rhs )
{
    int i = threadIdx.x;
    int j = blockIdx.y * gridDim.x + blockIdx.x;

    if( j >= n ){
        return;
    }
    int start = row[j];
    int end = row[j+1];
    int count = end-start;
    if( i == 0 ){
        sizes[j] = count;
        rhs[ j*WARP_SIZE+count-1 ] = MAGMA_C_ONE;
    }

    if ( i<count ){
        locations[ j*WARP_SIZE + i ] = col[ row[j]+i ];
    }
}// kernel

__global__ void
magma_clocations_trunc_upper_16kernel(
    magma_int_t n,
    magma_index_t *row,
    magma_index_t *col,
    magmaFloatComplex *val,
    magma_index_t *sizes,
    magma_index_t *locations,
    magmaFloatComplex *trisystems,
    magmaFloatComplex *rhs )
{
    int i = threadIdx.x;
    int j = blockIdx.y * gridDim.x + blockIdx.x;

    if( j >= n ){
        return;
    }
    int start = row[j];
    int end = row[j+1];
    int count = end-start;

    // normal case
    if( count <= BLOCKSIZE ){ // normal case
        if( i == 0 ){
            sizes[j] = count;
            rhs[ j*WARP_SIZE+count-1 ] = MAGMA_C_ONE;
        }
        if ( i<count ){
            locations[ j*WARP_SIZE + i ] = col[ row[j]+i ];
        }
    }
    else {
        // truncate in this row to the blocksize,
        // take only the 16 elements close to the main diagonal into account
        count = BLOCKSIZE;
        if (i == 0) {
            sizes[j] = count;
            rhs[ j*WARP_SIZE+count-1 ] = MAGMA_C_ONE;
        }

        locations[ j*WARP_SIZE + i ] = col[ row[j]+i ];
    }
}// kernel

__global__ void
magma_cfilltrisystems_16kernel(
    magma_int_t offset,
    magma_int_t limit,
    magma_index_t *row,
    magma_index_t *col,
    magmaFloatComplex *val,
    magma_index_t *sizes,
    magma_index_t *locations,
    magmaFloatComplex *trisystems,
    magmaFloatComplex *rhs )
{
    int i = (blockDim.x * blockIdx.x + threadIdx.x)+offset;
    int ii = (blockDim.x * blockIdx.x + threadIdx.x);

    if ( ii>=limit ){
        return;
    }
    //if ( i<offset ){
    //    return;
    //}

    for( int j=0; j<sizes[ i ]; j++ ){// no need for first
        int k = row[ locations[ j+i*WARP_SIZE ] ];
        int l = i*WARP_SIZE;
        int idx = 0;
        while( k < row[ locations[ j+i*WARP_SIZE ]+1 ] && l < (i+1)*WARP_SIZE ){ // stop once this column is done
            if( locations[ l ] == col[k] ){ //match
                // int loc = i*WARP_SIZE*WARP_SIZE + j*WARP_SIZE + idx;
                trisystems[ ii*WARP_SIZE*WARP_SIZE + j*WARP_SIZE + idx ]
                                                        = val[ k ];
                k++;
                l++;
                idx++;
            } else if( col[k] < locations[ l ] ){// need to check next element
                k++;
            } else { // element does not exist, i.e. l < LC.col[k]
                // printf("increment l\n");
                l++; // check next elment in the sparsity pattern
                idx++; // leave this element equal zero
            }
        }
    }
}// kernel


__global__ void
magma_cbackinsert_16kernel(
    magma_int_t n,
    magma_index_t *row,
    magma_index_t *col,
    magmaFloatComplex *val,
    magma_index_t *sizes,
    magmaFloatComplex *rhs )
{
    int i = threadIdx.x;
    int j = blockIdx.y * gridDim.x + blockIdx.x;
    int end = sizes[j];
    if( j >= n ){
        return;
    }

    if ( i>=end ){
        return;
    }

    val[row[j]+i] = rhs[j*WARP_SIZE+i];
}// kernel



#endif

/**
    Purpose
    -------
    This routine is designet to combine all kernels into one.

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
    L           magma_c_matrix
                triangular factor for which the ISAI matrix is computed.
                Col-Major CSR storage.

    @param[in,out]
    M           magma_c_matrix*
                SPAI preconditioner CSR col-major

    @param[out]
    sizes       magma_int_t*
                Number of Elements that are replaced.

    @param[out]
    locations   magma_int_t*
                Array indicating the locations.

    @param[out]
    trisystems  magmaFloatComplex*
                trisystems

    @param[out]
    rhs         magmaFloatComplex*
                right-hand sides

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_caux
    ********************************************************************/

extern "C" magma_int_t
magma_cisaigenerator_16_gpu(
    magma_uplo_t uplotype,
    magma_trans_t transtype,
    magma_diag_t diagtype,
    magma_c_matrix L,
    magma_c_matrix *M,
    magma_index_t *sizes,
    magma_index_t *locations,
    magmaFloatComplex *trisystems,
    magmaFloatComplex *rhs,
    magma_queue_t queue )
{
    magma_int_t info = 0;

#if (CUDA_VERSION >= 7000)
    magma_int_t arch = magma_getdevice_arch();

    cudaDeviceSetCacheConfig( cudaFuncCachePreferL1 );

    // routine 1
    int r1bs1 = WARP_SIZE;
    int r1bs2 = 1;
    int r1dg1 = min( int( sqrt( float( M->num_rows ))), 65535 );
    int r1dg2 = min(magma_ceildiv( M->num_rows, r1dg1 ), 65535);
    int r1dg3 = magma_ceildiv( M->num_rows, r1dg1*r1dg2 );

    dim3 r1block( r1bs1, r1bs2, 1 );
    dim3 r1grid( r1dg1, r1dg2, r1dg3 );

    int r2bs1 = WARP_SIZE;
    int r2bs2 = 1;
    int r2dg1 = magma_ceildiv( L.num_rows, r2bs1 );
    int r2dg2 = 1;
    int r2dg3 = 1;
    dim3 r2block( r2bs1, r2bs2, 1 );
    dim3 r2grid( r2dg1, r2dg2, r2dg3 );

    int r3bs1 = WARP_SIZE;
    int r3bs2 = 1;
    int r3dg1 = magma_ceildiv( 32000, r2bs1 );
    int r3dg2 = 1;
    int r3dg3 = 1;
    dim3 r3block( r3bs1, r3bs2, 1 );
    dim3 r3grid( r3dg1, r3dg2, r3dg3 );

    int recursive = magma_ceildiv( M->num_rows, 32000 );

    if (arch >= 300) {
        magma_cgpumemzero_16kernel<<< r1grid, r1block, 0, queue->cuda_stream() >>>(
                rhs, L.num_rows, WARP_SIZE, 1);

        if (uplotype == MagmaLower) {
            magma_clocations_lower_16kernel<<< r1grid, r1block, 0, queue->cuda_stream() >>>(
                            M->num_rows,
                            M->drow,
                            M->dcol,
                            M->dval,
                            sizes,
                            locations,
                            trisystems,
                            rhs );
        }
        else {
            magma_clocations_upper_16kernel<<< r1grid, r1block, 0, queue->cuda_stream() >>>(
                            M->num_rows,
                            M->drow,
                            M->dcol,
                            M->dval,
                            sizes,
                            locations,
                            trisystems,
                            rhs );
        }

        // chunk it recursively into batches of 1600
        for( int z=0; z<recursive; z++ ){
            int limit = min(32000, L.num_rows-32000*z);

            magma_cgpumemzero_16kernel<<< r1grid, r1block, 0, queue->cuda_stream() >>>(
                trisystems, limit, WARP_SIZE, WARP_SIZE );

            magma_cfilltrisystems_16kernel<<< r3grid, r3block, 0, queue->cuda_stream() >>>(
                                32000*z,
                                limit,
                                L.drow,
                                L.dcol,
                                L.dval,
                                sizes,
                                locations,
                                trisystems,
                                rhs );

            // routine 2
            if (uplotype == MagmaLower) {
                ctrsv_lower_16kernel_switch<<< r1grid, r1block, 0, queue->cuda_stream() >>>(
                        trisystems,
                        rhs+32000*16*z,
                        sizes+32000*z,
                        limit );
            }
            else {
                ctrsv_upper_16kernel_switch<<< r1grid, r1block, 0, queue->cuda_stream() >>>(
                        trisystems,
                        rhs+32000*16*z,
                        sizes+32000*z,
                        limit );
            }
        }

        // routine 3
        magma_cbackinsert_16kernel<<< r1grid, r1block, 0, queue->cuda_stream() >>>(
                M->num_rows,
                M->drow,
                M->dcol,
                M->dval,
                sizes,
                rhs );
    }
    else {
        info = MAGMA_ERR_NOT_SUPPORTED;
    }
#else
    // CUDA < 7000
    printf( "%% error: ISAI preconditioner requires CUDA > 6.0.\n" );
    info = MAGMA_ERR_NOT_SUPPORTED;
#endif

    return info;
}
