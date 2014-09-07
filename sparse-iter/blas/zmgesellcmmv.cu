/*
    -- MAGMA (version 1.5.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date September 2014

       @precisions normal z -> c d s

*/
#include "cuda_runtime.h"
#include <stdio.h>
#include "common_magma.h"
#include <cublas_v2.h>


#define PRECISION_z

#define TEXTURE


// SELLP SpMV kernel 3D grid
// see paper by M. KREUTZER, G. HAGER, G WELLEIN, H. FEHSKE A. BISHOP
// A UNIFIED SPARSE MATRIX DATA FORMAT 
// FOR MODERN PROCESSORS WITH WIDE SIMD UNITS
// SELLC SpMV kernel modified assigning multiple threads to each row - 2D kernel
__global__ void 
zmgesellptmv_kernel_1_3D( int num_rows, 
                     int num_cols,
                     int num_vecs,
                     int blocksize,
                     int T,
                     magmaDoubleComplex alpha, 
                     magmaDoubleComplex *d_val, 
                     magma_index_t *d_colind,
                     magma_index_t *d_rowptr,
                     magmaDoubleComplex *d_x,
                     magmaDoubleComplex beta, 
                     magmaDoubleComplex *d_y)
{
   // T threads assigned to each row
    int idx = threadIdx.x;      // local row
    int idy = threadIdx.y;      // vector
    int bdx = blockIdx.y * gridDim.x + blockIdx.x; // global block index
    int row = bdx * blocksize + idx;  // global row index


    if(row < num_rows ){
        magmaDoubleComplex dot = MAGMA_Z_MAKE(0.0, 0.0);
        int offset = d_rowptr[ bdx ];
        int max_ = (d_rowptr[ bdx+1 ]-offset)/blocksize;  
            // number of elements each thread handles

        for ( int k = 0; k < max_ ; k++ ){
            magmaDoubleComplex val = 
                        d_val[ offset + idx + blocksize*k ];
            int col = 
                    d_colind[ offset + idx + blocksize*k ] ;

            dot += val * d_x[ col*num_vecs+idy ];
        }
        d_y[ row+idy*num_rows ] = dot*alpha + beta*d_y [ row+idy*num_rows ];

    }

}


// SELLP SpMV kernel 3D grid
// see paper by M. KREUTZER, G. HAGER, G WELLEIN, H. FEHSKE A. BISHOP
// A UNIFIED SPARSE MATRIX DATA FORMAT 
// FOR MODERN PROCESSORS WITH WIDE SIMD UNITS
// SELLC SpMV kernel modified assigning multiple threads to each row - 2D kernel
__global__ void 
zmgesellptmv_kernel_4_3D( int num_rows, 
                     int num_cols,
                     int num_vecs,
                     int blocksize,
                     int T,
                     magmaDoubleComplex alpha, 
                     magmaDoubleComplex *d_val, 
                     magma_index_t *d_colind,
                     magma_index_t *d_rowptr,
                     magmaDoubleComplex *d_x,
                     magmaDoubleComplex beta, 
                     magmaDoubleComplex *d_y)
{
   // T threads assigned to each row
    int idx = threadIdx.y ;     // thread in row
    int idy = threadIdx.x;      // local row
    int idz = threadIdx.z;      // vector
    int ldx = idx * blocksize + idy;
    int ldz = idz * blocksize * T + idx * blocksize + idy;
    int bdx = blockIdx.y * gridDim.x + blockIdx.x; // global block index
    int row = bdx * blocksize + idy;  // global row index
    int vec = idz*num_rows;

    extern __shared__ magmaDoubleComplex shared[];


    if(row < num_rows ){
        magmaDoubleComplex dot = MAGMA_Z_MAKE(0.0, 0.0);
        int offset = d_rowptr[ bdx ];
        int block = blocksize * T; // total number of threads

        int max_ = (d_rowptr[ bdx+1 ]-offset)/block;  
            // number of elements each thread handles



        for ( int k = 0; k < max_ ; k++ ){
            magmaDoubleComplex val = 
                        d_val[ offset + ldx + block*k ];
            int col = 
                    d_colind[ offset + ldx + block*k ] ;

            dot += val * d_x[ col+vec ];
        }
        shared[ldz]  = dot;

        __syncthreads();
        if( idx < 2 ){
            shared[ldz]+=shared[ldz+blocksize*2];               
            __syncthreads();
            if( idx == 0 ) {
                d_y[row+vec] = 
                (shared[ldz]+shared[ldz+blocksize*1])*alpha 
                                            + beta*d_y [row+vec];
            }

        }

    }

}


// SELLP SpMV kernel 3D grid
// see paper by M. KREUTZER, G. HAGER, G WELLEIN, H. FEHSKE A. BISHOP
// A UNIFIED SPARSE MATRIX DATA FORMAT 
// FOR MODERN PROCESSORS WITH WIDE SIMD UNITS
// SELLC SpMV kernel modified assigning multiple threads to each row - 2D kernel
__global__ void 
zmgesellptmv_kernel_8_3D( int num_rows, 
                     int num_cols,
                     int num_vecs,
                     int blocksize,
                     int T,
                     magmaDoubleComplex alpha, 
                     magmaDoubleComplex *d_val, 
                     magma_index_t *d_colind,
                     magma_index_t *d_rowptr,
                     const magmaDoubleComplex* __restrict__ d_x,
                     magmaDoubleComplex beta, 
                     magmaDoubleComplex *d_y)
{
   // T threads assigned to each row
    int idx = threadIdx.y ;     // thread in row
    int idy = threadIdx.x;      // local row
    int idz = threadIdx.z;      // vector
    int ldx = idx * blocksize + idy;
    int ldz = idz * blocksize * T + idx * blocksize + idy;
    int bdx = blockIdx.y * gridDim.x + blockIdx.x; // global block index
    int row = bdx * blocksize + idy;  // global row index
    int vec = idz*num_rows;

    extern __shared__ magmaDoubleComplex shared[];


    if(row < num_rows ){
        magmaDoubleComplex dot = MAGMA_Z_MAKE(0.0, 0.0);
        int offset = d_rowptr[ bdx ];
        int block = blocksize * T; // total number of threads

        int max_ = (d_rowptr[ bdx+1 ]-offset)/block;  
            // number of elements each thread handles



        for ( int k = 0; k < max_ ; k++ ){
            magmaDoubleComplex val = 
                        d_val[ offset + ldx + block*k ];
            int col = 
                    d_colind[ offset + ldx + block*k ] ;

            dot += val * d_x[ col+vec ];
        }
        shared[ldz]  = dot;

        __syncthreads();
        if( idx < 4 ){
            shared[ldz]+=shared[ldz+blocksize*4];               
            __syncthreads();
            if( idx < 2 ) shared[ldz]+=shared[ldz+blocksize*2];   
            __syncthreads();
            if( idx == 0 ) {
                d_y[row+vec] = 
                (shared[ldz]+shared[ldz+blocksize*1])*alpha 
                                            + beta*d_y [row+vec];
            }

        }

    }

}


// SELLP SpMV kernel 3D grid
// see paper by M. KREUTZER, G. HAGER, G WELLEIN, H. FEHSKE A. BISHOP
// A UNIFIED SPARSE MATRIX DATA FORMAT 
// FOR MODERN PROCESSORS WITH WIDE SIMD UNITS
// SELLC SpMV kernel modified assigning multiple threads to each row - 2D kernel
__global__ void 
zmgesellptmv_kernel_16_3D( int num_rows, 
                     int num_cols,
                     int num_vecs,
                     int blocksize,
                     int T,
                     magmaDoubleComplex alpha, 
                     magmaDoubleComplex *d_val, 
                     magma_index_t *d_colind,
                     magma_index_t *d_rowptr,
                     magmaDoubleComplex *d_x,
                     magmaDoubleComplex beta, 
                     magmaDoubleComplex *d_y)
{
   // T threads assigned to each row
    int idx = threadIdx.y ;     // thread in row
    int idy = threadIdx.x;      // local row
    int idz = threadIdx.z;      // vector
    int ldx = idx * blocksize + idy;
    int ldz = idz * blocksize * T + idx * blocksize + idy;
    int bdx = blockIdx.y * gridDim.x + blockIdx.x; // global block index
    int row = bdx * blocksize + idy;  // global row index
    int vec = idz*num_rows;

    extern __shared__ magmaDoubleComplex shared[];


    if(row < num_rows ){
        magmaDoubleComplex dot = MAGMA_Z_MAKE(0.0, 0.0);
        int offset = d_rowptr[ bdx ];
        int block = blocksize * T; // total number of threads

        int max_ = (d_rowptr[ bdx+1 ]-offset)/block;  
            // number of elements each thread handles
        for ( int k = 0; k < max_ ; k++ ){
            magmaDoubleComplex val = 
                        d_val[ offset + ldx + block*k ];
            int col = 
                    d_colind[ offset + ldx + block*k ];

            dot += val * d_x[ col+vec ];
        }
        shared[ldz]  = dot;

        __syncthreads();
        if( idx < 8 ){
            shared[ldz]+=shared[ldz+blocksize*8];              
            __syncthreads();
            if( idx < 4 ) shared[ldz]+=shared[ldz+blocksize*4];   
            __syncthreads();
            if( idx < 2 ) shared[ldz]+=shared[ldz+blocksize*2];   
            __syncthreads();
            if( idx == 0 ) {
                d_y[row+vec] = 
                (shared[ldz]+shared[ldz+blocksize*1])*alpha 
                                            + beta*d_y [row+vec];
            }

        }

    }

}


// SELLP SpMV kernel 3D grid
// see paper by M. KREUTZER, G. HAGER, G WELLEIN, H. FEHSKE A. BISHOP
// A UNIFIED SPARSE MATRIX DATA FORMAT 
// FOR MODERN PROCESSORS WITH WIDE SIMD UNITS
// SELLC SpMV kernel modified assigning multiple threads to each row - 2D kernel
__global__ void 
zmgesellptmv_kernel_32_3D( int num_rows, 
                     int num_cols,
                     int num_vecs,
                     int blocksize,
                     int T,
                     magmaDoubleComplex alpha, 
                     magmaDoubleComplex *d_val, 
                     magma_index_t *d_colind,
                     magma_index_t *d_rowptr,
                     magmaDoubleComplex *d_x,
                     magmaDoubleComplex beta, 
                     magmaDoubleComplex *d_y)
{
   // T threads assigned to each row
    int idx = threadIdx.y ;     // thread in row
    int idy = threadIdx.x;      // local row
    int idz = threadIdx.z;      // vector
    int ldx = idx * blocksize + idy;
    int ldz = idz * blocksize * T + idx * blocksize + idy;
    int bdx = blockIdx.y * gridDim.x + blockIdx.x; // global block index
    int row = bdx * blocksize + idy;  // global row index
    int vec = idz*num_rows;

    extern __shared__ magmaDoubleComplex shared[];


    if(row < num_rows ){
        magmaDoubleComplex dot = MAGMA_Z_MAKE(0.0, 0.0);
        int offset = d_rowptr[ bdx ];
        int block = blocksize * T; // total number of threads

        int max_ = (d_rowptr[ bdx+1 ]-offset)/block;  
            // number of elements each thread handles
        for ( int k = 0; k < max_ ; k++ ){
            magmaDoubleComplex val = 
                        d_val[ offset + ldx + block*k ];
            int col = 
                    d_colind[ offset + ldx + block*k ];

            dot += val * d_x[ col+vec ];
        }
        shared[ldz]  = dot;

        __syncthreads();
        if( idx < 16 ){
            shared[ldz]+=shared[ldz+blocksize*16];              
            __syncthreads();
            if( idx < 8 ) shared[ldz]+=shared[ldz+blocksize*8];  
            __syncthreads();
            if( idx < 4 ) shared[ldz]+=shared[ldz+blocksize*4];   
            __syncthreads();
            if( idx < 2 ) shared[ldz]+=shared[ldz+blocksize*2];   
            __syncthreads();
            if( idx == 0 ) {
                d_y[row+vec] = 
                (shared[ldz]+shared[ldz+blocksize*1])*alpha 
                                            + beta*d_y [row+vec];
            }

        }

    }

}

/************************* same but using texture mem *************************/



// SELLP SpMV kernel 2D grid - for large number of vectors
// see paper by M. KREUTZER, G. HAGER, G WELLEIN, H. FEHSKE A. BISHOP
// A UNIFIED SPARSE MATRIX DATA FORMAT 
// FOR MODERN PROCESSORS WITH WIDE SIMD UNITS
// SELLC SpMV kernel modified assigning multiple threads to each row - 2D kernel
__global__ void 
zmgesellptmv_kernel_1_3D_tex( int num_rows, 
                     int num_cols,
                     int num_vecs,
                     int blocksize,
                     int T,
                     magmaDoubleComplex alpha, 
                     magmaDoubleComplex *d_val, 
                     magma_index_t *d_colind,
                     magma_index_t *d_rowptr,
                     cudaTextureObject_t texdx,
                     magmaDoubleComplex beta, 
                     magmaDoubleComplex *d_y)
{
#if defined(PRECISION_d) && defined(TEXTURE) && (__CUDA_ARCH__ >= 300)

    int idx = threadIdx.x;      // local row
    int idy = threadIdx.y;      // vector
    int bdx = blockIdx.y * gridDim.x + blockIdx.x; // global block index
    int row = bdx * blocksize + idx;  // global row index

    if(row < num_rows ){
        magmaDoubleComplex dot1 = MAGMA_Z_MAKE(0.0, 0.0);
        magmaDoubleComplex dot2 = MAGMA_Z_MAKE(0.0, 0.0);
        int offset = d_rowptr[ bdx ];
        int max_ = (d_rowptr[ bdx+1 ]-offset)/blocksize;  
            // number of elements each thread handles

        for ( int k = 0; k < max_ ; k++ ){
            magmaDoubleComplex val = 
                        d_val[ offset + idx + blocksize*k ];
            int col = 
                    num_vecs * d_colind[ offset + idx + blocksize*k ] ;

            int4 v = tex1Dfetch<int4>(texdx, col/2 + idy );
            dot1 += val * __hiloint2double(v.y, v.x);
            dot2 += val * __hiloint2double(v.w, v.z);
        }
        d_y[row+num_rows*idy*2] = 
                            dot1*alpha
                            + beta*d_y [row*num_vecs+idy*2];
        d_y[row+num_rows*idy*2+num_rows] = 
                            dot2*alpha
                            + beta*d_y [row*num_vecs+idy*2+1];
    }
#endif
}


// SELLP SpMV kernel 3D grid
// see paper by M. KREUTZER, G. HAGER, G WELLEIN, H. FEHSKE A. BISHOP
// A UNIFIED SPARSE MATRIX DATA FORMAT 
// FOR MODERN PROCESSORS WITH WIDE SIMD UNITS
// SELLC SpMV kernel modified assigning multiple threads to each row - 2D kernel
__global__ void 
zmgesellptmv_kernel_4_3D_tex( int num_rows, 
                     int num_cols,
                     int num_vecs,
                     int blocksize,
                     int T,
                     magmaDoubleComplex alpha, 
                     magmaDoubleComplex *d_val, 
                     magma_index_t *d_colind,
                     magma_index_t *d_rowptr,
                     cudaTextureObject_t texdx,
                     magmaDoubleComplex beta, 
                     magmaDoubleComplex *d_y)
{
#if defined(PRECISION_d) && defined(TEXTURE) && (__CUDA_ARCH__ >= 300)
   // T threads assigned to each row
    int idx = threadIdx.y ;     // thread in row
    int idy = threadIdx.x;      // local row
    int idz = threadIdx.z;      // vector
    int ldx = idx * blocksize + idy;
    int ldz = idz * blocksize * T + idx * blocksize + idy;
    int bdx = blockIdx.y * gridDim.x + blockIdx.x; // global block index
    int row = bdx * blocksize + idy;  // global row index
    int sv = num_vecs/2 * blocksize * T;

    extern __shared__ magmaDoubleComplex shared[];


    if(row < num_rows ){
        magmaDoubleComplex dot1 = MAGMA_Z_MAKE(0.0, 0.0);
        magmaDoubleComplex dot2 = MAGMA_Z_MAKE(0.0, 0.0);
        int offset = d_rowptr[ bdx ];
        int block = blocksize * T; // total number of threads

        int max_ = (d_rowptr[ bdx+1 ]-offset)/block;  
            // number of elements each thread handles



        for ( int k = 0; k < max_ ; k++ ){
            magmaDoubleComplex val = 
                        d_val[ offset + ldx + block*k ];
            int col = 
                    num_vecs * d_colind[ offset + ldx + block*k ] ;

            int4 v = tex1Dfetch<int4>(texdx, col/2 + idz );
            dot1 += val * __hiloint2double(v.y, v.x);
            dot2 += val * __hiloint2double(v.w, v.z);
        }
        shared[ldz]  = dot1;
        shared[ldz+sv]  = dot2;

        __syncthreads();
        if( idx < 2 ){
            shared[ldz]+=shared[ldz+blocksize*2];    
            shared[ldz+sv]+=shared[ldz+sv+blocksize*2];               
            __syncthreads();
            if( idx == 0 ) {
                d_y[row+num_rows*idz*2] = 
                (shared[ldz]+shared[ldz+blocksize*1])*alpha
                                            + beta*d_y [row*num_vecs+idz*2];
                d_y[row+num_rows*idz*2+num_rows] = 
                (shared[ldz+sv]+shared[ldz+sv+blocksize*1])*alpha
                                            + beta*d_y [row*num_vecs+idz*2+1];
            }

        }

    }
#endif
}
 

// SELLP SpMV kernel 3D grid
// see paper by M. KREUTZER, G. HAGER, G WELLEIN, H. FEHSKE A. BISHOP
// A UNIFIED SPARSE MATRIX DATA FORMAT 
// FOR MODERN PROCESSORS WITH WIDE SIMD UNITS
// SELLC SpMV kernel modified assigning multiple threads to each row - 2D kernel
__global__ void 
zmgesellptmv_kernel_8_3D_tex( int num_rows, 
                     int num_cols,
                     int num_vecs,
                     int blocksize,
                     int T,
                     magmaDoubleComplex alpha, 
                     magmaDoubleComplex *d_val, 
                     magma_index_t *d_colind,
                     magma_index_t *d_rowptr,
                     cudaTextureObject_t texdx,
                     magmaDoubleComplex beta, 
                     magmaDoubleComplex *d_y)
{
#if defined(PRECISION_d) && defined(TEXTURE) && (__CUDA_ARCH__ >= 300)
   // T threads assigned to each row
    int idx = threadIdx.y ;     // thread in row
    int idy = threadIdx.x;      // local row
    int idz = threadIdx.z;      // vector
    int ldx = idx * blocksize + idy;
    int ldz = idz * blocksize * T + idx * blocksize + idy;
    int bdx = blockIdx.y * gridDim.x + blockIdx.x; // global block index
    int row = bdx * blocksize + idy;  // global row index
    int sv = num_vecs/2 * blocksize * T;

    extern __shared__ magmaDoubleComplex shared[];


    if(row < num_rows ){
        magmaDoubleComplex dot1 = MAGMA_Z_MAKE(0.0, 0.0);
        magmaDoubleComplex dot2 = MAGMA_Z_MAKE(0.0, 0.0);
        int offset = d_rowptr[ bdx ];
        int block = blocksize * T; // total number of threads

        int max_ = (d_rowptr[ bdx+1 ]-offset)/block;  
            // number of elements each thread handles



        for ( int k = 0; k < max_ ; k++ ){
            magmaDoubleComplex val = 
                        d_val[ offset + ldx + block*k ];
            int col = 
                    num_vecs * d_colind[ offset + ldx + block*k ] ;

            int4 v = tex1Dfetch<int4>(texdx, col/2 + idz );
            dot1 += val * __hiloint2double(v.y, v.x);
            dot2 += val * __hiloint2double(v.w, v.z);
        }
        shared[ldz]  = dot1;
        shared[ldz+sv]  = dot2;

        __syncthreads();
        if( idx < 4 ){
            shared[ldz]+=shared[ldz+blocksize*4];    
            shared[ldz+sv]+=shared[ldz+sv+blocksize*4];               
            __syncthreads();
            if( idx < 2 ){
                shared[ldz]+=shared[ldz+blocksize*2];   
                shared[ldz+sv]+=shared[ldz+sv+blocksize*2];   
            }
            __syncthreads();
            if( idx == 0 ) {
                d_y[row+num_rows*idz*2] = 
                (shared[ldz]+shared[ldz+blocksize*1])*alpha
                                            + beta*d_y [row*num_vecs+idz*2];
                d_y[row+num_rows*idz*2+num_rows] = 
                (shared[ldz+sv]+shared[ldz+sv+blocksize*1])*alpha
                                            + beta*d_y [row*num_vecs+idz*2+1];
            }

        }

    }
#endif
}


// SELLP SpMV kernel 3D grid
// see paper by M. KREUTZER, G. HAGER, G WELLEIN, H. FEHSKE A. BISHOP
// A UNIFIED SPARSE MATRIX DATA FORMAT 
// FOR MODERN PROCESSORS WITH WIDE SIMD UNITS
// SELLC SpMV kernel modified assigning multiple threads to each row - 2D kernel
__global__ void 
zmgesellptmv_kernel_16_3D_tex( int num_rows, 
                     int num_cols,
                     int num_vecs,
                     int blocksize,
                     int T,
                     magmaDoubleComplex alpha, 
                     magmaDoubleComplex *d_val, 
                     magma_index_t *d_colind,
                     magma_index_t *d_rowptr,
                     cudaTextureObject_t texdx,
                     magmaDoubleComplex beta, 
                     magmaDoubleComplex *d_y)
{
#if defined(PRECISION_d) && defined(TEXTURE) && (__CUDA_ARCH__ >= 300)
   // T threads assigned to each row
    int idx = threadIdx.y ;     // thread in row
    int idy = threadIdx.x;      // local row
    int idz = threadIdx.z;      // vector
    int ldx = idx * blocksize + idy;
    int ldz = idz * blocksize * T + idx * blocksize + idy;
    int bdx = blockIdx.y * gridDim.x + blockIdx.x; // global block index
    int row = bdx * blocksize + idy;  // global row index
    int sv = num_vecs/2 * blocksize * T;

    extern __shared__ magmaDoubleComplex shared[];


    if(row < num_rows ){
        magmaDoubleComplex dot1 = MAGMA_Z_MAKE(0.0, 0.0);
        magmaDoubleComplex dot2 = MAGMA_Z_MAKE(0.0, 0.0);
        int offset = d_rowptr[ bdx ];
        int block = blocksize * T; // total number of threads

        int max_ = (d_rowptr[ bdx+1 ]-offset)/block;  
            // number of elements each thread handles



        for ( int k = 0; k < max_ ; k++ ){
            magmaDoubleComplex val = 
                        d_val[ offset + ldx + block*k ];
            int col = 
                    num_vecs * d_colind[ offset + ldx + block*k ] ;

            int4 v = tex1Dfetch<int4>(texdx, col/2 + idz );
            dot1 += val * __hiloint2double(v.y, v.x);
            dot2 += val * __hiloint2double(v.w, v.z);
        }
        shared[ldz]  = dot1;
        shared[ldz+sv]  = dot2;

        __syncthreads();
        if( idx < 8 ){
            shared[ldz]+=shared[ldz+blocksize*8];    
            shared[ldz+sv]+=shared[ldz+sv+blocksize*8];               
            __syncthreads();
            if( idx < 4 ){
                shared[ldz]+=shared[ldz+blocksize*4];   
                shared[ldz+sv]+=shared[ldz+sv+blocksize*4];   
            }
            if( idx < 2 ){
                shared[ldz]+=shared[ldz+blocksize*2];   
                shared[ldz+sv]+=shared[ldz+sv+blocksize*2];   
            }
            __syncthreads();
            if( idx == 0 ) {
                d_y[row+num_rows*idz*2] = 
                (shared[ldz]+shared[ldz+blocksize*1])*alpha
                                            + beta*d_y [row*num_vecs+idz*2];
                d_y[row+num_rows*idz*2+num_rows] = 
                (shared[ldz+sv]+shared[ldz+sv+blocksize*1])*alpha
                                            + beta*d_y [row*num_vecs+idz*2+1];
            }

        }

    }
#endif
}


// SELLP SpMV kernel 3D grid
// see paper by M. KREUTZER, G. HAGER, G WELLEIN, H. FEHSKE A. BISHOP
// A UNIFIED SPARSE MATRIX DATA FORMAT 
// FOR MODERN PROCESSORS WITH WIDE SIMD UNITS
// SELLC SpMV kernel modified assigning multiple threads to each row - 2D kernel
__global__ void 
zmgesellptmv_kernel_32_3D_tex( int num_rows, 
                     int num_cols,
                     int num_vecs,
                     int blocksize,
                     int T,
                     magmaDoubleComplex alpha, 
                     magmaDoubleComplex *d_val, 
                     magma_index_t *d_colind,
                     magma_index_t *d_rowptr,
                     cudaTextureObject_t texdx,
                     magmaDoubleComplex beta, 
                     magmaDoubleComplex *d_y)
{
#if defined(PRECISION_d) && defined(TEXTURE) && (__CUDA_ARCH__ >= 300)
   // T threads assigned to each row
    int idx = threadIdx.y ;     // thread in row
    int idy = threadIdx.x;      // local row
    int idz = threadIdx.z;      // vector
    int ldx = idx * blocksize + idy;
    int ldz = idz * blocksize * T + idx * blocksize + idy;
    int bdx = blockIdx.y * gridDim.x + blockIdx.x; // global block index
    int row = bdx * blocksize + idy;  // global row index
    int sv = num_vecs/2 * blocksize * T;

    extern __shared__ magmaDoubleComplex shared[];


    if(row < num_rows ){
        magmaDoubleComplex dot1 = MAGMA_Z_MAKE(0.0, 0.0);
        magmaDoubleComplex dot2 = MAGMA_Z_MAKE(0.0, 0.0);
        int offset = d_rowptr[ bdx ];
        int block = blocksize * T; // total number of threads

        int max_ = (d_rowptr[ bdx+1 ]-offset)/block;  
            // number of elements each thread handles



        for ( int k = 0; k < max_ ; k++ ){
            magmaDoubleComplex val = 
                        d_val[ offset + ldx + block*k ];
            int col = 
                    num_vecs * d_colind[ offset + ldx + block*k ] ;

            int4 v = tex1Dfetch<int4>(texdx, col/2 + idz );
            dot1 += val * __hiloint2double(v.y, v.x);
            dot2 += val * __hiloint2double(v.w, v.z);
        }
        shared[ldz]  = dot1;
        shared[ldz+sv]  = dot2;

        __syncthreads();
        if( idx < 16 ){
            shared[ldz]+=shared[ldz+blocksize*16];    
            shared[ldz+sv]+=shared[ldz+sv+blocksize*16];               
            __syncthreads();
            if( idx < 8 ){
                shared[ldz]+=shared[ldz+blocksize*8];   
                shared[ldz+sv]+=shared[ldz+sv+blocksize*8];   
            }
            if( idx < 4 ){
                shared[ldz]+=shared[ldz+blocksize*4];   
                shared[ldz+sv]+=shared[ldz+sv+blocksize*4];   
            }
            if( idx < 2 ){
                shared[ldz]+=shared[ldz+blocksize*2];   
                shared[ldz+sv]+=shared[ldz+sv+blocksize*2];   
            }
            __syncthreads();
            if( idx == 0 ) {
                d_y[row+num_rows*idz*2] = 
                (shared[ldz]+shared[ldz+blocksize*1])*alpha
                                            + beta*d_y [row*num_vecs+idz*2];
                d_y[row+num_rows*idz*2+num_rows] = 
                (shared[ldz+sv]+shared[ldz+sv+blocksize*1])*alpha
                                            + beta*d_y [row*num_vecs+idz*2+1];
            }

        }

    }
#endif
}

//***************** routines for beta = 0 ************************************//


// SELLP SpMV kernel 2D grid - for large number of vectors
// see paper by M. KREUTZER, G. HAGER, G WELLEIN, H. FEHSKE A. BISHOP
// A UNIFIED SPARSE MATRIX DATA FORMAT 
// FOR MODERN PROCESSORS WITH WIDE SIMD UNITS
// SELLC SpMV kernel modified assigning multiple threads to each row - 2D kernel
__global__ void 
zmgesellptmv_kernel_1_3D_texb( int num_rows, 
                     int num_cols,
                     int num_vecs,
                     int blocksize,
                     int T,
                     magmaDoubleComplex alpha, 
                     magmaDoubleComplex *d_val, 
                     magma_index_t *d_colind,
                     magma_index_t *d_rowptr,
                     cudaTextureObject_t texdx,
                     magmaDoubleComplex *d_y)
{
#if defined(PRECISION_d) && defined(TEXTURE) && (__CUDA_ARCH__ >= 300)

    int idx = threadIdx.x;      // local row
    int idy = threadIdx.y;      // vector
    int bdx = blockIdx.y * gridDim.x + blockIdx.x; // global block index
    int row = bdx * blocksize + idx;  // global row index

    if(row < num_rows ){
        magmaDoubleComplex dot1 = MAGMA_Z_MAKE(0.0, 0.0);
        magmaDoubleComplex dot2 = MAGMA_Z_MAKE(0.0, 0.0);
        int offset = d_rowptr[ bdx ];
        int block = blocksize * T; // total number of threads

        int max_ = (d_rowptr[ bdx+1 ]-offset)/block;  
            // number of elements each thread handles

        for ( int k = 0; k < max_ ; k++ ){
            magmaDoubleComplex val = 
                        d_val[ offset + idx + blocksize*k ];
            int col = 
                    num_vecs * d_colind[ offset + idx + blocksize*k ] ;

            int4 v = tex1Dfetch<int4>(texdx, col/2 + idy );
            dot1 += val * __hiloint2double(v.y, v.x);
            dot2 += val * __hiloint2double(v.w, v.z);
        }
        d_y[row+num_rows*idy*2] = 
                            dot1*alpha;
        d_y[row+num_rows*idy*2+num_rows] = 
                            dot2*alpha;
    }
#endif
}


// SELLP SpMV kernel 3D grid
// see paper by M. KREUTZER, G. HAGER, G WELLEIN, H. FEHSKE A. BISHOP
// A UNIFIED SPARSE MATRIX DATA FORMAT 
// FOR MODERN PROCESSORS WITH WIDE SIMD UNITS
// SELLC SpMV kernel modified assigning multiple threads to each row - 2D kernel
__global__ void 
zmgesellptmv_kernel_4_3D_texb( int num_rows, 
                     int num_cols,
                     int num_vecs,
                     int blocksize,
                     int T,
                     magmaDoubleComplex alpha, 
                     magmaDoubleComplex *d_val, 
                     magma_index_t *d_colind,
                     magma_index_t *d_rowptr,
                     cudaTextureObject_t texdx,
                     magmaDoubleComplex *d_y)
{
#if defined(PRECISION_d) && defined(TEXTURE) && (__CUDA_ARCH__ >= 300)
   // T threads assigned to each row
    int idx = threadIdx.y ;     // thread in row
    int idy = threadIdx.x;      // local row
    int idz = threadIdx.z;      // vector
    int ldx = idx * blocksize + idy;
    int ldz = idz * blocksize * T + idx * blocksize + idy;
    int bdx = blockIdx.y * gridDim.x + blockIdx.x; // global block index
    int row = bdx * blocksize + idy;  // global row index
    int sv = num_vecs/2 * blocksize * T;

    extern __shared__ magmaDoubleComplex shared[];


    if(row < num_rows ){
        magmaDoubleComplex dot1 = MAGMA_Z_MAKE(0.0, 0.0);
        magmaDoubleComplex dot2 = MAGMA_Z_MAKE(0.0, 0.0);
        int offset = d_rowptr[ bdx ];
        int block = blocksize * T; // total number of threads

        int max_ = (d_rowptr[ bdx+1 ]-offset)/block;  
            // number of elements each thread handles



        for ( int k = 0; k < max_ ; k++ ){
            magmaDoubleComplex val = 
                        d_val[ offset + ldx + block*k ];
            int col = 
                    num_vecs * d_colind[ offset + ldx + block*k ] ;

            int4 v = tex1Dfetch<int4>(texdx, col/2 + idz );
            dot1 += val * __hiloint2double(v.y, v.x);
            dot2 += val * __hiloint2double(v.w, v.z);
        }
        shared[ldz]  = dot1;
        shared[ldz+sv]  = dot2;

        __syncthreads();
        if( idx < 2 ){
            shared[ldz]+=shared[ldz+blocksize*2];    
            shared[ldz+sv]+=shared[ldz+sv+blocksize*2];               
            __syncthreads();
            if( idx == 0 ) {
                d_y[row+num_rows*idz*2] = 
                (shared[ldz]+shared[ldz+blocksize*1])*alpha;
                d_y[row+num_rows*idz*2+num_rows] = 
                (shared[ldz+sv]+shared[ldz+sv+blocksize*1])*alpha;
            }

        }

    }
#endif
}
 

// SELLP SpMV kernel 3D grid
// see paper by M. KREUTZER, G. HAGER, G WELLEIN, H. FEHSKE A. BISHOP
// A UNIFIED SPARSE MATRIX DATA FORMAT 
// FOR MODERN PROCESSORS WITH WIDE SIMD UNITS
// SELLC SpMV kernel modified assigning multiple threads to each row - 2D kernel
__global__ void 
zmgesellptmv_kernel_8_3D_texb( int num_rows, 
                     int num_cols,
                     int num_vecs,
                     int blocksize,
                     int T,
                     magmaDoubleComplex alpha, 
                     magmaDoubleComplex *d_val, 
                     magma_index_t *d_colind,
                     magma_index_t *d_rowptr,
                     cudaTextureObject_t texdx,
                     magmaDoubleComplex *d_y)
{
#if defined(PRECISION_d) && defined(TEXTURE) && (__CUDA_ARCH__ >= 300)
   // T threads assigned to each row
    int idx = threadIdx.y ;     // thread in row
    int idy = threadIdx.x;      // local row
    int idz = threadIdx.z;      // vector
    int ldx = idx * blocksize + idy;
    int ldz = idz * blocksize * T + idx * blocksize + idy;
    int bdx = blockIdx.y * gridDim.x + blockIdx.x; // global block index
    int row = bdx * blocksize + idy;  // global row index
    int sv = num_vecs/2 * blocksize * T;

    extern __shared__ magmaDoubleComplex shared[];


    if(row < num_rows ){
        magmaDoubleComplex dot1 = MAGMA_Z_MAKE(0.0, 0.0);
        magmaDoubleComplex dot2 = MAGMA_Z_MAKE(0.0, 0.0);
        int offset = d_rowptr[ bdx ];
        int block = blocksize * T; // total number of threads

        int max_ = (d_rowptr[ bdx+1 ]-offset)/block;  
            // number of elements each thread handles



        for ( int k = 0; k < max_ ; k++ ){
            magmaDoubleComplex val = 
                        d_val[ offset + ldx + block*k ];
            int col = 
                    num_vecs * d_colind[ offset + ldx + block*k ] ;

            int4 v = tex1Dfetch<int4>(texdx, col/2 + idz );
            dot1 += val * __hiloint2double(v.y, v.x);
            dot2 += val * __hiloint2double(v.w, v.z);
        }
        shared[ldz]  = dot1;
        shared[ldz+sv]  = dot2;

        __syncthreads();
        if( idx < 4 ){
            shared[ldz]+=shared[ldz+blocksize*4];    
            shared[ldz+sv]+=shared[ldz+sv+blocksize*4];               
            __syncthreads();
            if( idx < 2 ){
                shared[ldz]+=shared[ldz+blocksize*2];   
                shared[ldz+sv]+=shared[ldz+sv+blocksize*2];   
            }
            __syncthreads();
            if( idx == 0 ) {
                d_y[row+num_rows*idz*2] = 
                (shared[ldz]+shared[ldz+blocksize*1])*alpha;

                d_y[row+num_rows*idz*2+num_rows] = 
                (shared[ldz+sv]+shared[ldz+sv+blocksize*1])*alpha;
            }

        }

    }
#endif
}


// SELLP SpMV kernel 3D grid
// see paper by M. KREUTZER, G. HAGER, G WELLEIN, H. FEHSKE A. BISHOP
// A UNIFIED SPARSE MATRIX DATA FORMAT 
// FOR MODERN PROCESSORS WITH WIDE SIMD UNITS
// SELLC SpMV kernel modified assigning multiple threads to each row - 2D kernel
__global__ void 
zmgesellptmv_kernel_16_3D_texb( int num_rows, 
                     int num_cols,
                     int num_vecs,
                     int blocksize,
                     int T,
                     magmaDoubleComplex alpha, 
                     magmaDoubleComplex *d_val, 
                     magma_index_t *d_colind,
                     magma_index_t *d_rowptr,
                     cudaTextureObject_t texdx,
                     magmaDoubleComplex *d_y)
{
#if defined(PRECISION_d) && defined(TEXTURE) && (__CUDA_ARCH__ >= 300)
   // T threads assigned to each row
    int idx = threadIdx.y ;     // thread in row
    int idy = threadIdx.x;      // local row
    int idz = threadIdx.z;      // vector
    int ldx = idx * blocksize + idy;
    int ldz = idz * blocksize * T + idx * blocksize + idy;
    int bdx = blockIdx.y * gridDim.x + blockIdx.x; // global block index
    int row = bdx * blocksize + idy;  // global row index
    int sv = num_vecs/2 * blocksize * T;

    extern __shared__ magmaDoubleComplex shared[];


    if(row < num_rows ){
        magmaDoubleComplex dot1 = MAGMA_Z_MAKE(0.0, 0.0);
        magmaDoubleComplex dot2 = MAGMA_Z_MAKE(0.0, 0.0);
        int offset = d_rowptr[ bdx ];
        int block = blocksize * T; // total number of threads

        int max_ = (d_rowptr[ bdx+1 ]-offset)/block;  
            // number of elements each thread handles



        for ( int k = 0; k < max_ ; k++ ){
            magmaDoubleComplex val = 
                        d_val[ offset + ldx + block*k ];
            int col = 
                    num_vecs * d_colind[ offset + ldx + block*k ] ;

            int4 v = tex1Dfetch<int4>(texdx, col/2 + idz );
            dot1 += val * __hiloint2double(v.y, v.x);
            dot2 += val * __hiloint2double(v.w, v.z);
        }
        shared[ldz]  = dot1;
        shared[ldz+sv]  = dot2;

        __syncthreads();
        if( idx < 8 ){
            shared[ldz]+=shared[ldz+blocksize*8];    
            shared[ldz+sv]+=shared[ldz+sv+blocksize*8];               
            __syncthreads();
            if( idx < 4 ){
                shared[ldz]+=shared[ldz+blocksize*4];   
                shared[ldz+sv]+=shared[ldz+sv+blocksize*4];   
            }
            if( idx < 2 ){
                shared[ldz]+=shared[ldz+blocksize*2];   
                shared[ldz+sv]+=shared[ldz+sv+blocksize*2];   
            }
            __syncthreads();
            if( idx == 0 ) {
                d_y[row+num_rows*idz*2] = 
                (shared[ldz]+shared[ldz+blocksize*1])*alpha;

                d_y[row+num_rows*idz*2+num_rows] = 
                (shared[ldz+sv]+shared[ldz+sv+blocksize*1])*alpha;
            }

        }

    }
#endif
}


// SELLP SpMV kernel 3D grid
// see paper by M. KREUTZER, G. HAGER, G WELLEIN, H. FEHSKE A. BISHOP
// A UNIFIED SPARSE MATRIX DATA FORMAT 
// FOR MODERN PROCESSORS WITH WIDE SIMD UNITS
// SELLC SpMV kernel modified assigning multiple threads to each row - 2D kernel
__global__ void 
zmgesellptmv_kernel_32_3D_texb( int num_rows, 
                     int num_cols,
                     int num_vecs,
                     int blocksize,
                     int T,
                     magmaDoubleComplex alpha, 
                     magmaDoubleComplex *d_val, 
                     magma_index_t *d_colind,
                     magma_index_t *d_rowptr,
                     cudaTextureObject_t texdx,
                     magmaDoubleComplex *d_y)
{
#if defined(PRECISION_d) && defined(TEXTURE) && (__CUDA_ARCH__ >= 300)
   // T threads assigned to each row
    int idx = threadIdx.y ;     // thread in row
    int idy = threadIdx.x;      // local row
    int idz = threadIdx.z;      // vector
    int ldx = idx * blocksize + idy;
    int ldz = idz * blocksize * T + idx * blocksize + idy;
    int bdx = blockIdx.y * gridDim.x + blockIdx.x; // global block index
    int row = bdx * blocksize + idy;  // global row index
    int sv = num_vecs/2 * blocksize * T;

    extern __shared__ magmaDoubleComplex shared[];


    if(row < num_rows ){
        magmaDoubleComplex dot1 = MAGMA_Z_MAKE(0.0, 0.0);
        magmaDoubleComplex dot2 = MAGMA_Z_MAKE(0.0, 0.0);
        int offset = d_rowptr[ bdx ];
        int block = blocksize * T; // total number of threads

        int max_ = (d_rowptr[ bdx+1 ]-offset)/block;  
            // number of elements each thread handles



        for ( int k = 0; k < max_ ; k++ ){
            magmaDoubleComplex val = 
                        d_val[ offset + ldx + block*k ];
            int col = 
                    num_vecs * d_colind[ offset + ldx + block*k ] ;

            int4 v = tex1Dfetch<int4>(texdx, col/2 + idz );
            dot1 += val * __hiloint2double(v.y, v.x);
            dot2 += val * __hiloint2double(v.w, v.z);
        }
        shared[ldz]  = dot1;
        shared[ldz+sv]  = dot2;

        __syncthreads();
        if( idx < 16 ){
            shared[ldz]+=shared[ldz+blocksize*16];    
            shared[ldz+sv]+=shared[ldz+sv+blocksize*16];               
            __syncthreads();
            if( idx < 8 ){
                shared[ldz]+=shared[ldz+blocksize*8];   
                shared[ldz+sv]+=shared[ldz+sv+blocksize*8];   
            }
            if( idx < 4 ){
                shared[ldz]+=shared[ldz+blocksize*4];   
                shared[ldz+sv]+=shared[ldz+sv+blocksize*4];   
            }
            if( idx < 2 ){
                shared[ldz]+=shared[ldz+blocksize*2];   
                shared[ldz+sv]+=shared[ldz+sv+blocksize*2];   
            }
            __syncthreads();
            if( idx == 0 ) {
                d_y[row+num_rows*idz*2] = 
                (shared[ldz]+shared[ldz+blocksize*1])*alpha;

                d_y[row+num_rows*idz*2+num_rows] = 
                (shared[ldz+sv]+shared[ldz+sv+blocksize*1])*alpha;
            }

        }

    }
#endif
}


//*************************** end  kernels using texture  ********************//



/**
    Purpose
    -------
    
    This routine computes Y = alpha *  A^t *  X + beta * Y on the GPU.
    Input format is SELLP. Note, that the input format for X is row-major
    while the output format for Y is column major!
    
    Arguments
    ---------

    @param
    transA      magma_trans_t
                transpose A?

    @param
    m           magma_int_t
                number of rows in A

    @param
    n           magma_int_t
                number of columns in A 

    @param
    num_vecs    magma_int_t
                number of columns in X and Y

    @param
    blocksize   magma_int_t
                number of rows in one ELL-slice

    @param
    slices      magma_int_t
                number of slices in matrix

    @param
    alignment   magma_int_t
                number of threads assigned to one row

    @param
    alpha       magmaDoubleComplex
                scalar multiplier

    @param
    d_val       magmaDoubleComplex*
                array containing values of A in SELLP

    @param
    d_colind    magma_int_t*
                columnindices of A in SELLP

    @param
    d_rowptr    magma_int_t*
                rowpointer of SELLP

    @param
    d_x         magmaDoubleComplex*
                input vector x

    @param
    beta        magmaDoubleComplex
                scalar multiplier

    @param
    d_y         magmaDoubleComplex*
                input/output vector y


    @ingroup magmasparse_zblas
    ********************************************************************/

extern "C" magma_int_t
magma_zmgesellpmv( magma_trans_t transA,
                    magma_int_t m, magma_int_t n,
                    magma_int_t num_vecs,
                    magma_int_t blocksize,
                    magma_int_t slices,
                    magma_int_t alignment,
                    magmaDoubleComplex alpha,
                    magmaDoubleComplex *d_val,
                    magma_index_t *d_colind,
                    magma_index_t *d_rowptr,
                    magmaDoubleComplex *d_x,
                    magmaDoubleComplex beta,
                    magmaDoubleComplex *d_y ){

    // using a 3D thread grid for small num_vecs, a 2D grid otherwise
    
    int texture=0, kepler=0, precision=0;

    magma_int_t arch = magma_getdevice_arch();
    if ( arch > 300 )
        kepler = 1;
           
    #if defined(PRECISION_d)
        precision = 1;
    #endif

    #if defined(TEXTURE)
        texture = 1;
    #endif

    if( (texture==1) && (precision==1) && (kepler==1) ){

        // Create channel.
        cudaChannelFormatDesc channel_desc;
        channel_desc = cudaCreateChannelDesc(32, 32, 32, 32, 
                                        cudaChannelFormatKindSigned);

        // Create resource descriptor.
        struct cudaResourceDesc resDescdx;
        memset(&resDescdx, 0, sizeof(resDescdx));
        resDescdx.resType = cudaResourceTypeLinear;
        resDescdx.res.linear.devPtr = (void*)d_x;
        resDescdx.res.linear.desc = channel_desc;
        resDescdx.res.linear.sizeInBytes = m * num_vecs * sizeof(double);

        // Specify texture object parameters.
        struct cudaTextureDesc texDesc;
        memset(&texDesc, 0, sizeof(texDesc));
        texDesc.addressMode[0] = cudaAddressModeClamp;
        texDesc.filterMode     = cudaFilterModePoint;
        texDesc.readMode       = cudaReadModeElementType;

        // Create texture object.
        cudaTextureObject_t texdx = 0;
        cudaCreateTextureObject(&texdx, &resDescdx, &texDesc, NULL);

        cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);

        if( num_vecs%2 ==1 ){ // only multiple of 2 can be processed
            printf("error: number of vectors has to be multiple of 2.\n");
            exit(-1);
        }
        if( num_vecs > 8 ) // avoid running into memory problems
            alignment = 1; 

        int num_threads = (num_vecs/2) * blocksize*alignment;            

        // every thread handles two vectors
        if (  num_threads > 1024 )
            printf("error: too many threads requested.\n");

        dim3 block( blocksize, alignment, num_vecs/2 );

        int dimgrid1 = sqrt(slices);
        int dimgrid2 = (slices + dimgrid1 -1 ) / dimgrid1;

        dim3 grid( dimgrid1, dimgrid2, 1);
        int Ms = num_vecs * blocksize*alignment * sizeof( magmaDoubleComplex );


        if( alignment == 1){
            dim3 block( blocksize, num_vecs/2, 1 );
            if( beta == MAGMA_Z_MAKE( 0.0, 0.0 ) )
            zmgesellptmv_kernel_1_3D_texb<<< grid, block, 0, magma_stream >>>
            ( m, n, num_vecs, blocksize, alignment, alpha,
                d_val, d_colind, d_rowptr, texdx, d_y );
            else
            zmgesellptmv_kernel_1_3D_tex<<< grid, block, 0, magma_stream >>>
            ( m, n, num_vecs, blocksize, alignment, alpha,
                d_val, d_colind, d_rowptr, texdx, beta, d_y );
        }
        else if( alignment == 4){
            dim3 block( blocksize, alignment, num_vecs/2 );
            if( beta == MAGMA_Z_MAKE( 0.0, 0.0 ) )
            zmgesellptmv_kernel_4_3D_texb<<< grid, block, Ms, magma_stream >>>
            ( m, n, num_vecs, blocksize, alignment, alpha,
                d_val, d_colind, d_rowptr, texdx, d_y );
            else
            zmgesellptmv_kernel_4_3D_tex<<< grid, block, Ms, magma_stream >>>
            ( m, n, num_vecs, blocksize, alignment, alpha,
                d_val, d_colind, d_rowptr, texdx, beta, d_y );
        }
        else if( alignment == 8){
            dim3 block( blocksize, alignment, num_vecs/2 );
            if( beta == MAGMA_Z_MAKE( 0.0, 0.0 ) )
            zmgesellptmv_kernel_8_3D_texb<<< grid, block, Ms, magma_stream >>>
            ( m, n, num_vecs, blocksize, alignment, alpha,
                d_val, d_colind, d_rowptr, texdx, d_y );
            else
            zmgesellptmv_kernel_8_3D_tex<<< grid, block, Ms, magma_stream >>>
            ( m, n, num_vecs, blocksize, alignment, alpha,
                d_val, d_colind, d_rowptr, texdx, beta, d_y );
        }
        else if( alignment == 16){
            dim3 block( blocksize, alignment, num_vecs/2 );
            if( beta == MAGMA_Z_MAKE( 0.0, 0.0 ) )
            zmgesellptmv_kernel_16_3D_texb<<< grid, block, Ms, magma_stream >>>
            ( m, n, num_vecs, blocksize, alignment, alpha,
                d_val, d_colind, d_rowptr, texdx, d_y );
            else
            zmgesellptmv_kernel_16_3D_tex<<< grid, block, Ms, magma_stream >>>
            ( m, n, num_vecs, blocksize, alignment, alpha,
                d_val, d_colind, d_rowptr, texdx, beta, d_y );
        }
        else if( alignment == 32){
            dim3 block( blocksize, alignment, num_vecs/2 );
            if( beta == MAGMA_Z_MAKE( 0.0, 0.0 ) )
            zmgesellptmv_kernel_32_3D_texb<<< grid, block, Ms, magma_stream >>>
            ( m, n, num_vecs, blocksize, alignment, alpha,
                d_val, d_colind, d_rowptr, texdx, d_y );
            else
            zmgesellptmv_kernel_32_3D_tex<<< grid, block, Ms, magma_stream >>>
            ( m, n, num_vecs, blocksize, alignment, alpha,
                d_val, d_colind, d_rowptr, texdx, beta, d_y );
        }
        else{
            printf("error: alignment %d not supported.\n", alignment);
            exit(-1);
        }

    }else{

        if( num_vecs%2 ==1 ){ // only multiple of 2 can be processed
            printf("error: number of vectors has to be multiple of 2.\n");
            exit(-1);
        }

        if( num_vecs > 8 ) // avoid running into memory problems
            alignment = 1;

        int num_threads = num_vecs * blocksize*alignment;

        // every thread handles two vectors
        if (  num_threads > 1024 )
            printf("error: too many threads requested.\n");

        int dimgrid1 = sqrt(slices);
        int dimgrid2 = (slices + dimgrid1 -1 ) / dimgrid1;

        dim3 grid( dimgrid1, dimgrid2, 1);
        int Ms =  num_threads * sizeof( magmaDoubleComplex );

        if( alignment == 1){
            dim3 block( blocksize, num_vecs, 1 ); 
            zmgesellptmv_kernel_1_3D<<< grid, block, 0, magma_stream >>>
            ( m, n, num_vecs, blocksize, alignment, alpha,
                d_val, d_colind, d_rowptr, d_x, beta, d_y );
        }
        else if( alignment == 4){
            dim3 block( blocksize, alignment, num_vecs );
            zmgesellptmv_kernel_4_3D<<< grid, block, Ms, magma_stream >>>
            ( m, n, num_vecs, blocksize, alignment, alpha,
                d_val, d_colind, d_rowptr, d_x, beta, d_y );
        }
        else if( alignment == 8){
            dim3 block( blocksize, alignment, num_vecs );
            zmgesellptmv_kernel_8_3D<<< grid, block, Ms, magma_stream >>>
            ( m, n, num_vecs, blocksize, alignment, alpha,
                d_val, d_colind, d_rowptr, d_x, beta, d_y );
        }
        else if( alignment == 16){
            dim3 block( blocksize, alignment, num_vecs );
            zmgesellptmv_kernel_16_3D<<< grid, block, Ms, magma_stream >>>
            ( m, n, num_vecs, blocksize, alignment, alpha,
                d_val, d_colind, d_rowptr, d_x, beta, d_y );
        }
        else if( alignment == 32){
            dim3 block( blocksize, alignment, num_vecs );
            zmgesellptmv_kernel_32_3D<<< grid, block, Ms, magma_stream >>>
            ( m, n, num_vecs, blocksize, alignment, alpha,
                d_val, d_colind, d_rowptr, d_x, beta, d_y );
        }
        else{
            printf("error: alignment %d not supported.\n", alignment);
            exit(-1);
        }
    }

   return MAGMA_SUCCESS;
}

