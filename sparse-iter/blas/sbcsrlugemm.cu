/*
    -- MAGMA (version 1.5.0-beta2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date May 2014

       @generated from zbcsrlugemm.cu normal z -> s, Fri May 30 10:41:36 2014

*/
#include <stdio.h>
#include <cuda_runtime_api.h>
#include <cublas_v2.h>  // include before magma.h

#include "magma.h"


#if (GPUSHMEM < 200)
   #define BLOCK_SIZE 128
#else
   #define BLOCK_SIZE 512
#endif

#define PRECISION_s

#define  Ablockinfo(i,j)  Ablockinfo[(i)*c_blocks   + (j)]
#define  Bblockinfo(i,j)  Bblockinfo[(i)*c_blocks   + (j)]
#define A(i,j) ((Ablockinfo(i,j)-1)*size_b*size_b)
#define B(i,j) ((Bblockinfo(i,j)-1)*size_b*size_b)

//============================================================

#define ldb m
#define lda m
#define ldc m


#define fetch_x_A(i) (((i)<m*m)?Aval[i]:0)
#define fetch_x_B(i) (((i)<m*m)?B[i]:0)


// every multiprocessor handles one BCSR-block
__global__ void 
sbcsr_gemm_kernel32( 
                  int m,
                  int n,
                  int kblocks,   
                  float **Avals, 
                  float **Bval,
                  float **Cval)
{
#if (__CUDA_ARCH__ >= 200)

#if defined(PRECISION_d)
    const  int tx = threadIdx.x;
    const  int ty = threadIdx.y;
  
    const int idt = ty * 64 + tx;

    const int tx2 = idt%16;
    const int ty2 = idt/16;

    float xxB[4];
    float *B;

    int trackA = __mul24( ty2, lda) + tx2 ;
    float *Aval = Avals[blockIdx.z];

    __shared__ float Abs[64][65];
    __shared__ float  Bb[16][65];


    for(int j=ty2; j<64; j+=16){
        for(int y=tx2; y<64; y+=16){
           Abs[y][j] = fetch_x_A(trackA + y-tx2) ;
            }
        trackA += __mul24( 16, m);
    }

    for(int k=0; k<kblocks; k++){
        B = Bval[k];
        int trackB = tx2+ __mul24( ty2 * 16, ldb );

        // Prefetch part of B
          #pragma unroll
          for(int y=0; y<4; y++){
                 Bb[tx2][ty2*4+y] = fetch_x_B( trackB + y * ldb) ;
          }
        __syncthreads();    // this is necessary!!!

        float Axs[4];
        float Bxp[4];
        float Cb[16] = {0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0};

        int k1;
        for(k1=0; k1<m-16; k1+=16)
        {
                trackB += 16;

                #pragma unroll
                for( int y=0; y<4; y++)
                        xxB[y] = fetch_x_B( trackB + y*ldb);
                #pragma unroll
                for( int j1=0;j1<16;j1++)
                {
                        #pragma unroll
                        for( int y=0; y<4; y++){
                                Axs[y] =  Abs[tx2+y*16][j1+k1] ;
                        }

                        #pragma unroll
                        for( int y=0; y<4; y++){
                                Bxp[y]= Bb[j1][ty2+y*16];
                        }

                        #pragma unroll
                        for( int x=0; x<4; x++)
                        {
                                #pragma unroll
                                for( int y=0; y<4; y++)
                                {
                                        Cb[x*4+y]  += Axs[x]*Bxp[y];
                                }
                        }

                }
                #pragma unroll
                for(int y=0; y<4; y++)
                        Bb[tx2][ty2*4 + y] = xxB[y];

                __syncthreads();     // this is necessary!!!
        }
        // Prepare where to write the result
        float *C = Cval[blockIdx.z * kblocks + k];
        C += tx2 + __mul24 (ty2 ,ldc);

        #pragma unroll
        for(int j1=0;j1<16;j1++)
        {

                #pragma unroll
                for( int y=0; y<4; y++)
                        Axs[y] =  Abs[tx2 + y*16][j1+k1] ;

                #pragma unroll
                for( int y=0; y<4; y++)
                        Bxp[y]= Bb[j1][ty2 + y*16];

                #pragma unroll
                for( int x=0; x<4; x++)
                {
                        #pragma unroll
                        for( int y=0;y<4; y++)
                        {
                                Cb[x*4 + y]  += Axs[x]*Bxp[y];
                        }
                }
        }   
        int gy = ty2;
        #pragma unroll
        for( int y=0;y<4;y++, gy+=16)
        {
                int gx = tx2;
        #pragma unroll
                for(int x=0;x<4;x++, gx+=16)
                {
                        if (gx < m && gy < n){
                              C[x*16] -= Cb[y+x*4];
                       }
                }
                C += ldc*16;
        }
      }
#endif

#endif
}

// every multiprocessor handles one BCSR-block
__global__ void 
sbcsr_gemm_kernel64( 
                  int m,
                  int n,
                  int kblocks,   
                  float **Avals, 
                  float **Bval,
                  float **Cval)
{
#if (__CUDA_ARCH__ >= 200)

#if defined(PRECISION_d)
    const  int tx = threadIdx.x;
    const  int ty = threadIdx.y;
  
    const int idt = ty * 64 + tx;

    const int tx2 = idt%16;
    const int ty2 = idt/16;

    float xxB[4];

    float *B;

    int trackA = __mul24( ty2, lda) + tx2 ;
    float *Aval = Avals[blockIdx.z];

    __shared__ float Abs[64][65];
    __shared__ float  Bb[16][65];


    for(int j=ty2; j<64; j+=16){
        for(int y=tx2; y<64; y+=16){
           Abs[y][j] = fetch_x_A(trackA + y-tx2) ;
            }
        trackA += __mul24( 16, m);
    }


    for(int k=0; k<kblocks; k++){

        B = Bval[k];
        int trackB = tx2+ __mul24( ty2 * 4, ldb );

        // Prefetch part of B
          #pragma unroll
          for(int y=0; y<4; y++){
                 Bb[tx2][ty2*4+y] = fetch_x_B( trackB + y * ldb) ;
          }

        __syncthreads();    // this is necessary!!!

        float Axs[4];
        float Bxp[4];

        float Cb[16] = {0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0};

        int k1;
        for(k1=0; k1<m-16; k1+=16)
        {
                trackB += 16;

                #pragma unroll
                for( int y=0; y<4; y++)
                        xxB[y] = fetch_x_B( trackB + y*ldb);

                #pragma unroll
                for( int j1=0;j1<16;j1++)
                {
                        #pragma unroll
                        for( int y=0; y<4; y++){
                                Axs[y] =  Abs[tx2+y*16][j1+k1] ;
                        }

                        #pragma unroll
                        for( int y=0; y<4; y++){
                                Bxp[y]= Bb[j1][ty2+y*16];
                        }

                        #pragma unroll
                        for( int x=0; x<4; x++)
                        {
                                #pragma unroll
                                for( int y=0; y<4; y++)
                                {
                                        Cb[x*4+y]  += Axs[x]*Bxp[y];
                                }
                        }

                }

                __syncthreads();
                #pragma unroll
                for(int y=0; y<4; y++)
                        Bb[tx2][ty2*4 + y] = xxB[y];

                __syncthreads();     // this is necessary!!!

        }
        // Prepare where to write the result
        float *C = Cval[blockIdx.z * kblocks + k];
        C += tx2 + __mul24 (ty2 ,ldc);

        #pragma unroll
        for(int j1=0;j1<16;j1++)
        {

                #pragma unroll
                for( int y=0; y<4; y++)
                        Axs[y] =  Abs[tx2 + y*16][j1+k1] ;

                #pragma unroll
                for( int y=0; y<4; y++)
                        Bxp[y]= Bb[j1][ty2 + y*16];

                #pragma unroll
                for( int x=0; x<4; x++)
                {
                        #pragma unroll
                        for( int y=0;y<4; y++)
                        {
                                Cb[x*4 + y]  += Axs[x]*Bxp[y];
                        }
                }
        }   

        int gy = ty2;
        #pragma unroll
        for( int y=0;y<4;y++, gy+=16)
        {
                int gx = tx2;
        #pragma unroll
                for(int x=0;x<4;x++, gx+=16)
                {
                        if (gx < m && gy < n){
                              C[x*16] -= Cb[y+x*4];
                       }
                }

                C += ldc*16;
        }

      }
#endif

#endif
}





/*  -- MAGMA (version 1.5.0-beta2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date May 2014

    Purpose
    =======
    
    For a Block-CSR ILU factorization, this routine updates all blocks in
    the trailing matrix.
    
    Arguments
    =========

    magma_int_t size_b              blocksize in BCSR
    magma_int_t num_block_rows      number of block rows
    magma_int_t kblocks             number of blocks in row
    magma_int_t *ipiv               array containing pivots
    float *x           input/output vector x

    ======================================================================    */

extern "C" magma_int_t
magma_sbcsrluegemm( magma_int_t size_b, 
                    magma_int_t num_block_rows,
                    magma_int_t kblocks,
                    float **dA,  
                    float **dB,  
                    float **dC ){

#if defined(PRECISION_d)

    magma_int_t arch = magma_getdevice_arch();

    if ( arch < 200  ) {
        printf("error: magma_sbcsrluegemm needs a CUDA architecture"
               " with at least 48K shared memory (Fermi +).\n"
               "Please run sbcsrlu.cpp using CUBLAS batched.\n");
    
    }
    else {

    dim3 threads( 64, 4 );

    dim3 grid(1, 1, num_block_rows);
    sbcsr_gemm_kernel64<<< grid, threads, 0, magma_stream >>>( 
                  size_b, size_b, kblocks, dA, dB, dC );

    }

#else
    printf("error: currently only supported for real.\n"
           "Please run sbcsrlu.cpp using CUBLAS batched.\n");
#endif

    return MAGMA_SUCCESS;
}



