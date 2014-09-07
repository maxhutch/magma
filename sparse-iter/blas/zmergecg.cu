/*
    -- MAGMA (version 1.5.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date September 2014

       @precisions normal z -> c d s
       @author Hartwig Anzt

*/
#include "common_magma.h"
#include "magmasparse.h"

#define BLOCK_SIZE 512

#define PRECISION_z


// These routines merge multiple kernels from zmergecg into one
// for a description see 
// "Reformulated Conjugate Gradient for the Energy-Aware 
// Solution of Linear Systems on GPUs (ICPP '13)

// accelerated reduction for one vector
__global__ void 
magma_zcgreduce_kernel_spmv1( int Gs,
                           int n, 
                           magmaDoubleComplex *vtmp,
                           magmaDoubleComplex *vtmp2 ){

    extern __shared__ magmaDoubleComplex temp[];    
    int Idx = threadIdx.x;
    int blockSize = 128;
    int gridSize = blockSize  * 2 * gridDim.x; 
    temp[Idx] = MAGMA_Z_MAKE( 0.0, 0.0);
    int i = blockIdx.x * ( blockSize * 2 ) + Idx;   
    while (i < Gs ) {
        temp[ Idx  ] += vtmp[ i ]; 
        temp[ Idx  ] += ( i + blockSize < Gs ) ? vtmp[ i + blockSize ] 
                                                : MAGMA_Z_MAKE( 0.0, 0.0); 
        i += gridSize;
    }
    __syncthreads();
    if ( Idx < 64 ){
        temp[ Idx ] += temp[ Idx + 64 ];
    }
    __syncthreads();
    #if defined(PRECISION_z) || defined(PRECISION_c)
        if( Idx < 32 ){
            temp[ Idx ] += temp[ Idx + 32 ];__syncthreads();
            temp[ Idx ] += temp[ Idx + 16 ];__syncthreads();
            temp[ Idx ] += temp[ Idx + 8 ];__syncthreads();
            temp[ Idx ] += temp[ Idx + 4 ];__syncthreads();
            temp[ Idx ] += temp[ Idx + 2 ];__syncthreads();
            temp[ Idx ] += temp[ Idx + 1 ];__syncthreads();
        }
    #endif
    #if defined(PRECISION_d)
        if( Idx < 32 ){
            volatile double *temp2 = temp;
            temp2[ Idx ] += temp2[ Idx + 32 ];
            temp2[ Idx ] += temp2[ Idx + 16 ];
            temp2[ Idx ] += temp2[ Idx + 8 ];
            temp2[ Idx ] += temp2[ Idx + 4 ];
            temp2[ Idx ] += temp2[ Idx + 2 ];
            temp2[ Idx ] += temp2[ Idx + 1 ];
        }
    #endif
    #if defined(PRECISION_s)
        if( Idx < 32 ){
            volatile float *temp2 = temp;
            temp2[ Idx ] += temp2[ Idx + 32 ];
            temp2[ Idx ] += temp2[ Idx + 16 ];
            temp2[ Idx ] += temp2[ Idx + 8 ];
            temp2[ Idx ] += temp2[ Idx + 4 ];
            temp2[ Idx ] += temp2[ Idx + 2 ];
            temp2[ Idx ] += temp2[ Idx + 1 ];
        }
    #endif
    if ( Idx == 0 ){
        vtmp2[ blockIdx.x ] = temp[ 0 ];
    }
}

// computes the SpMV using CSR and the first step of the reduction
__global__ void 
magma_zcgmerge_spmvcsr_kernel(  
                 int n,
                 magmaDoubleComplex *d_val, 
                 magma_index_t *d_rowptr, 
                 magma_index_t *d_colind,
                 magmaDoubleComplex *d,
                 magmaDoubleComplex *z,
                 magmaDoubleComplex *vtmp
                                           ){

    extern __shared__ magmaDoubleComplex temp[]; 
    int Idx = threadIdx.x;   
    int i   = blockIdx.x * blockDim.x + Idx;
    int j;

    temp[ Idx ] = MAGMA_Z_MAKE( 0.0, 0.0);

    if( i<n ){
        magmaDoubleComplex dot = MAGMA_Z_ZERO;
        int start = d_rowptr[ i ];
        int end = d_rowptr[ i+1 ];
        for( j=start; j<end; j++)
            dot += d_val[ j ] * d[ d_colind[j] ];
        z[ i ] =  dot;
        temp[ Idx ] =  d[ i ] * dot;
    }

    __syncthreads();
    if ( Idx < 128 ){
        temp[ Idx ] += temp[ Idx + 128 ];
    }
    __syncthreads();
    if ( Idx < 64 ){
        temp[ Idx ] += temp[ Idx + 64 ];
    }
    __syncthreads();
    #if defined(PRECISION_z) || defined(PRECISION_c)
        if( Idx < 32 ){
            temp[ Idx ] += temp[ Idx + 32 ];__syncthreads();
            temp[ Idx ] += temp[ Idx + 16 ];__syncthreads();
            temp[ Idx ] += temp[ Idx + 8 ];__syncthreads();
            temp[ Idx ] += temp[ Idx + 4 ];__syncthreads();
            temp[ Idx ] += temp[ Idx + 2 ];__syncthreads();
            temp[ Idx ] += temp[ Idx + 1 ];__syncthreads();
        }
    #endif
    #if defined(PRECISION_d)
        if( Idx < 32 ){
            volatile double *temp2 = temp;
            temp2[ Idx ] += temp2[ Idx + 32 ];
            temp2[ Idx ] += temp2[ Idx + 16 ];
            temp2[ Idx ] += temp2[ Idx + 8 ];
            temp2[ Idx ] += temp2[ Idx + 4 ];
            temp2[ Idx ] += temp2[ Idx + 2 ];
            temp2[ Idx ] += temp2[ Idx + 1 ];
        }
    #endif
    #if defined(PRECISION_s)
        if( Idx < 32 ){
            volatile float *temp2 = temp;
            temp2[ Idx ] += temp2[ Idx + 32 ];
            temp2[ Idx ] += temp2[ Idx + 16 ];
            temp2[ Idx ] += temp2[ Idx + 8 ];
            temp2[ Idx ] += temp2[ Idx + 4 ];
            temp2[ Idx ] += temp2[ Idx + 2 ];
            temp2[ Idx ] += temp2[ Idx + 1 ];
        }
    #endif

    if ( Idx == 0 ){
            vtmp[ blockIdx.x ] = temp[ 0 ];
    }
}

// computes the SpMV using ELL and the first step of the reduction
__global__ void 
magma_zcgmerge_spmvellpackt_kernel(  
                 int n,
                 int num_cols_per_row,
                 magmaDoubleComplex *d_val, 
                 magma_index_t *d_colind,
                 magmaDoubleComplex *d,
                 magmaDoubleComplex *z,
                 magmaDoubleComplex *vtmp
                                           ){

    extern __shared__ magmaDoubleComplex temp[]; 
    int Idx = threadIdx.x;   
    int i   = blockIdx.x * blockDim.x + Idx;

    temp[ Idx ] = MAGMA_Z_MAKE( 0.0, 0.0);

    if(i < n ){
        magmaDoubleComplex dot = MAGMA_Z_MAKE(0.0, 0.0);
        for ( int k = 0; k < num_cols_per_row ; k ++){
            int col = d_colind [ n * k + i ];
            magmaDoubleComplex val = d_val [ n * k + i ];
            if( val != 0)
                dot += val * d[ col ];
        }
        z[ i ] =  dot;
        temp[ Idx ] = d[ i ] * dot;
    }

    __syncthreads();
    if ( Idx < 128 ){
        temp[ Idx ] += temp[ Idx + 128 ];
    }
    __syncthreads();
    if ( Idx < 64 ){
        temp[ Idx ] += temp[ Idx + 64 ];
    }
    __syncthreads();
    #if defined(PRECISION_z) || defined(PRECISION_c)
        if( Idx < 32 ){
            temp[ Idx ] += temp[ Idx + 32 ];__syncthreads();
            temp[ Idx ] += temp[ Idx + 16 ];__syncthreads();
            temp[ Idx ] += temp[ Idx + 8 ];__syncthreads();
            temp[ Idx ] += temp[ Idx + 4 ];__syncthreads();
            temp[ Idx ] += temp[ Idx + 2 ];__syncthreads();
            temp[ Idx ] += temp[ Idx + 1 ];__syncthreads();
        }
    #endif
    #if defined(PRECISION_d)
        if( Idx < 32 ){
            volatile double *temp2 = temp;
            temp2[ Idx ] += temp2[ Idx + 32 ];
            temp2[ Idx ] += temp2[ Idx + 16 ];
            temp2[ Idx ] += temp2[ Idx + 8 ];
            temp2[ Idx ] += temp2[ Idx + 4 ];
            temp2[ Idx ] += temp2[ Idx + 2 ];
            temp2[ Idx ] += temp2[ Idx + 1 ];
        }
    #endif
    #if defined(PRECISION_s)
        if( Idx < 32 ){
            volatile float *temp2 = temp;
            temp2[ Idx ] += temp2[ Idx + 32 ];
            temp2[ Idx ] += temp2[ Idx + 16 ];
            temp2[ Idx ] += temp2[ Idx + 8 ];
            temp2[ Idx ] += temp2[ Idx + 4 ];
            temp2[ Idx ] += temp2[ Idx + 2 ];
            temp2[ Idx ] += temp2[ Idx + 1 ];
        }
    #endif

    if ( Idx == 0 ){
            vtmp[ blockIdx.x ] = temp[ 0 ];
    }
}

// computes the SpMV using ELLPACK and the first step of the reduction
__global__ void 
magma_zcgmerge_spmvellpack_kernel(  
                 int n,
                 int num_cols_per_row,
                 magmaDoubleComplex *d_val, 
                 magma_index_t *d_colind,
                 magmaDoubleComplex *d,
                 magmaDoubleComplex *z,
                 magmaDoubleComplex *vtmp
                                           ){

    extern __shared__ magmaDoubleComplex temp[]; 
    int Idx = threadIdx.x;   
    int i   = blockIdx.x * blockDim.x + Idx;

    temp[ Idx ] = MAGMA_Z_MAKE( 0.0, 0.0);

    if(i < n ){
        magmaDoubleComplex dot = MAGMA_Z_MAKE(0.0, 0.0);
        for ( int k = 0; k < num_cols_per_row ; k ++){
            int col = d_colind [ num_cols_per_row * i + k ];
            magmaDoubleComplex val = d_val [ num_cols_per_row * i + k ];
            if( val != 0)
                dot += val * d[ col ];
        }
        z[ i ] =  dot;
        temp[ Idx ] = d[ i ] * dot;
    }

    __syncthreads();
    if ( Idx < 128 ){
        temp[ Idx ] += temp[ Idx + 128 ];
    }
    __syncthreads();
    if ( Idx < 64 ){
        temp[ Idx ] += temp[ Idx + 64 ];
    }
    __syncthreads();
    #if defined(PRECISION_z) || defined(PRECISION_c)
        if( Idx < 32 ){
            temp[ Idx ] += temp[ Idx + 32 ];__syncthreads();
            temp[ Idx ] += temp[ Idx + 16 ];__syncthreads();
            temp[ Idx ] += temp[ Idx + 8 ];__syncthreads();
            temp[ Idx ] += temp[ Idx + 4 ];__syncthreads();
            temp[ Idx ] += temp[ Idx + 2 ];__syncthreads();
            temp[ Idx ] += temp[ Idx + 1 ];__syncthreads();
        }
    #endif
    #if defined(PRECISION_d)
        if( Idx < 32 ){
            volatile double *temp2 = temp;
            temp2[ Idx ] += temp2[ Idx + 32 ];
            temp2[ Idx ] += temp2[ Idx + 16 ];
            temp2[ Idx ] += temp2[ Idx + 8 ];
            temp2[ Idx ] += temp2[ Idx + 4 ];
            temp2[ Idx ] += temp2[ Idx + 2 ];
            temp2[ Idx ] += temp2[ Idx + 1 ];
        }
    #endif
    #if defined(PRECISION_s)
        if( Idx < 32 ){
            volatile float *temp2 = temp;
            temp2[ Idx ] += temp2[ Idx + 32 ];
            temp2[ Idx ] += temp2[ Idx + 16 ];
            temp2[ Idx ] += temp2[ Idx + 8 ];
            temp2[ Idx ] += temp2[ Idx + 4 ];
            temp2[ Idx ] += temp2[ Idx + 2 ];
            temp2[ Idx ] += temp2[ Idx + 1 ];
        }
    #endif

    if ( Idx == 0 ){
            vtmp[ blockIdx.x ] = temp[ 0 ];
    }
}

// computes the SpMV using ELLRT 8 threads per row
__global__ void 
magma_zcgmerge_spmvellpackrt_kernel_8(  
                 int n,
                 magmaDoubleComplex *d_val, 
                 magma_index_t *d_colind,
                 magma_index_t *d_rowlength,
                 magmaDoubleComplex *d,
                 magmaDoubleComplex *z,
                 magmaDoubleComplex *vtmp,
                 magma_int_t T, 
                 magma_int_t alignment  ){

int idx = blockIdx.y * gridDim.x * blockDim.x + 
          blockDim.x * blockIdx.x + threadIdx.x ; // global thread index
int idb = threadIdx.x ;  // local thread index
int idp = idb%T;  // number of threads assigned to one row
int i = idx/T;  // row index

extern __shared__ magmaDoubleComplex shared[];

    if(i < n ){
        magmaDoubleComplex dot = MAGMA_Z_MAKE(0.0, 0.0);
        int max_ = (d_rowlength[i]+T-1)/T;  
            // number of elements each thread handles

        for ( int k = 0; k < max_ ; k++ ){

            // original code in paper (not working for me)
            //magmaDoubleComplex val = d_val[ k*(T*alignment)+(i*T)+idp ];  
            //int col = d_colind [ k*(T*alignment)+(i*T)+idp ];    

            // new code (working for me)        
            magmaDoubleComplex val = d_val[ k*(T)+(i*alignment)+idp ];
            int col = d_colind [ k*(T)+(i*alignment)+idp ];

            dot += val * d[ col ];
        }
        shared[idb]  = dot;
        if( idp < 4 ){
            shared[idb]+=shared[idb+4];
            if( idp < 2 ) shared[idb]+=shared[idb+2];
            if( idp == 0 ) {
                z[i] = (shared[idb]+shared[idb+1]);
            }

        }
    }

}

// computes the SpMV using ELLRT 8 threads per row
__global__ void 
magma_zcgmerge_spmvellpackrt_kernel_16(  
                 int n,
                 magmaDoubleComplex *d_val, 
                 magma_index_t *d_colind,
                 magma_index_t *d_rowlength,
                 magmaDoubleComplex *d,
                 magmaDoubleComplex *z,
                 magmaDoubleComplex *vtmp,
                 magma_int_t T, 
                 magma_int_t alignment  ){

int idx = blockIdx.y * gridDim.x * blockDim.x + 
          blockDim.x * blockIdx.x + threadIdx.x ; // global thread index
int idb = threadIdx.x ;  // local thread index
int idp = idb%T;  // number of threads assigned to one row
int i = idx/T;  // row index

extern __shared__ magmaDoubleComplex shared[];

    if(i < n ){
        magmaDoubleComplex dot = MAGMA_Z_MAKE(0.0, 0.0);
        int max_ = (d_rowlength[i]+T-1)/T;  
            // number of elements each thread handles

        for ( int k = 0; k < max_ ; k++ ){

            // original code in paper (not working for me)
            //magmaDoubleComplex val = d_val[ k*(T*alignment)+(i*T)+idp ];  
            //int col = d_colind [ k*(T*alignment)+(i*T)+idp ];    

            // new code (working for me)        
            magmaDoubleComplex val = d_val[ k*(T)+(i*alignment)+idp ];
            int col = d_colind [ k*(T)+(i*alignment)+idp ];

            dot += val * d[ col ];
        }
        shared[idb]  = dot;
        if( idp < 8 ){
            shared[idb]+=shared[idb+8];
            if( idp < 4 ) shared[idb]+=shared[idb+4];
            if( idp < 2 ) shared[idb]+=shared[idb+2];
            if( idp == 0 ) {
                z[i] = (shared[idb]+shared[idb+1]);
            }

        }
    }

}

// computes the SpMV using ELLRT 8 threads per row
__global__ void 
magma_zcgmerge_spmvellpackrt_kernel_32(  
                 int n,
                 magmaDoubleComplex *d_val, 
                 magma_index_t *d_colind,
                 magma_index_t *d_rowlength,
                 magmaDoubleComplex *d,
                 magmaDoubleComplex *z,
                 magmaDoubleComplex *vtmp,
                 magma_int_t T, 
                 magma_int_t alignment  ){

int idx = blockIdx.y * gridDim.x * blockDim.x + 
          blockDim.x * blockIdx.x + threadIdx.x ; // global thread index
int idb = threadIdx.x ;  // local thread index
int idp = idb%T;  // number of threads assigned to one row
int i = idx/T;  // row index

extern __shared__ magmaDoubleComplex shared[];

    if(i < n ){
        magmaDoubleComplex dot = MAGMA_Z_MAKE(0.0, 0.0);
        int max_ = (d_rowlength[i]+T-1)/T;  
            // number of elements each thread handles

        for ( int k = 0; k < max_ ; k++ ){

            // original code in paper (not working for me)
            //magmaDoubleComplex val = d_val[ k*(T*alignment)+(i*T)+idp ];  
            //int col = d_colind [ k*(T*alignment)+(i*T)+idp ];    

            // new code (working for me)        
            magmaDoubleComplex val = d_val[ k*(T)+(i*alignment)+idp ];
            int col = d_colind [ k*(T)+(i*alignment)+idp ];

            dot += val * d[ col ];
        }
        shared[idb]  = dot;
        if( idp < 16 ){
            shared[idb]+=shared[idb+16];
            if( idp < 8 ) shared[idb]+=shared[idb+8];
            if( idp < 4 ) shared[idb]+=shared[idb+4];
            if( idp < 2 ) shared[idb]+=shared[idb+2];
            if( idp == 0 ) {
                z[i] = (shared[idb]+shared[idb+1]);
            }

        }
    }

}





// additional kernel necessary to compute first reduction step
__global__ void 
magma_zcgmerge_spmvellpackrt_kernel2(  
                 int n,
                 magmaDoubleComplex *z,
                 magmaDoubleComplex *d,
                 magmaDoubleComplex *vtmp2
                                           ){

    extern __shared__ magmaDoubleComplex temp[]; 
    int Idx = threadIdx.x;   
    int i   = blockIdx.x * blockDim.x + Idx;
    

    temp[ Idx ] = ( i < n ) ? z[i]*d[i] : MAGMA_Z_MAKE(0.0, 0.0);
    __syncthreads();
    if ( Idx < 128 ){
        temp[ Idx ] += temp[ Idx + 128 ];
    }
    __syncthreads();
    if ( Idx < 64 ){
        temp[ Idx ] += temp[ Idx + 64 ];
    }
    __syncthreads();
    #if defined(PRECISION_z) || defined(PRECISION_c)
        if( Idx < 32 ){
            temp[ Idx ] += temp[ Idx + 32 ];__syncthreads();
            temp[ Idx ] += temp[ Idx + 16 ];__syncthreads();
            temp[ Idx ] += temp[ Idx + 8 ];__syncthreads();
            temp[ Idx ] += temp[ Idx + 4 ];__syncthreads();
            temp[ Idx ] += temp[ Idx + 2 ];__syncthreads();
            temp[ Idx ] += temp[ Idx + 1 ];__syncthreads();
        }
    #endif
    #if defined(PRECISION_d)
        if( Idx < 32 ){
            volatile double *temp2 = temp;
            temp2[ Idx ] += temp2[ Idx + 32 ];
            temp2[ Idx ] += temp2[ Idx + 16 ];
            temp2[ Idx ] += temp2[ Idx + 8 ];
            temp2[ Idx ] += temp2[ Idx + 4 ];
            temp2[ Idx ] += temp2[ Idx + 2 ];
            temp2[ Idx ] += temp2[ Idx + 1 ];
        }
    #endif
    #if defined(PRECISION_s)
        if( Idx < 32 ){
            volatile float *temp2 = temp;
            temp2[ Idx ] += temp2[ Idx + 32 ];
            temp2[ Idx ] += temp2[ Idx + 16 ];
            temp2[ Idx ] += temp2[ Idx + 8 ];
            temp2[ Idx ] += temp2[ Idx + 4 ];
            temp2[ Idx ] += temp2[ Idx + 2 ];
            temp2[ Idx ] += temp2[ Idx + 1 ];
        }
    #endif

    if ( Idx == 0 ){
            vtmp2[ blockIdx.x ] = temp[ 0 ];
    }
}



// computes the SpMV using SELLC
__global__ void 
magma_zcgmerge_spmvsellc_kernel(   
                     int num_rows, 
                     int blocksize,
                     magmaDoubleComplex *d_val, 
                     magma_index_t *d_colind,
                     magma_index_t *d_rowptr,
                     magmaDoubleComplex *d,
                     magmaDoubleComplex *z,
                     magmaDoubleComplex *vtmp){

    extern __shared__ magmaDoubleComplex temp[]; 
    int Idx = threadIdx.x;   
    int i   = blockIdx.x * blockDim.x + Idx;
    int offset = d_rowptr[ blockIdx.x ];
    int border = (d_rowptr[ blockIdx.x+1 ]-offset)/blocksize;

 temp[ Idx ] = MAGMA_Z_MAKE( 0.0, 0.0);


    if(i < num_rows ){
        magmaDoubleComplex dot = MAGMA_Z_MAKE(0.0, 0.0);
        for ( int n = 0; n < border; n ++){
            int col = d_colind [offset+ blocksize * n + Idx ];
            magmaDoubleComplex val = d_val[offset+ blocksize * n + Idx];
            if( val != 0){
                  dot=dot+val*d[col];
            }
        }
        z[ i ] =  dot;
        temp[ Idx ] = d[ i ] * dot;
    }
    __syncthreads();
    if ( Idx < 128 ){
        temp[ Idx ] += temp[ Idx + 128 ];
    }
    __syncthreads();
    if ( Idx < 64 ){
        temp[ Idx ] += temp[ Idx + 64 ];
    }
    __syncthreads();
    #if defined(PRECISION_z) || defined(PRECISION_c)
        if( Idx < 32 ){
            temp[ Idx ] += temp[ Idx + 32 ];__syncthreads();
            temp[ Idx ] += temp[ Idx + 16 ];__syncthreads();
            temp[ Idx ] += temp[ Idx + 8 ];__syncthreads();
            temp[ Idx ] += temp[ Idx + 4 ];__syncthreads();
            temp[ Idx ] += temp[ Idx + 2 ];__syncthreads();
            temp[ Idx ] += temp[ Idx + 1 ];__syncthreads();
        }
    #endif
    #if defined(PRECISION_d)
        if( Idx < 32 ){
            volatile double *temp2 = temp;
            temp2[ Idx ] += temp2[ Idx + 32 ];
            temp2[ Idx ] += temp2[ Idx + 16 ];
            temp2[ Idx ] += temp2[ Idx + 8 ];
            temp2[ Idx ] += temp2[ Idx + 4 ];
            temp2[ Idx ] += temp2[ Idx + 2 ];
            temp2[ Idx ] += temp2[ Idx + 1 ];
        }
    #endif
    #if defined(PRECISION_s)
        if( Idx < 32 ){
            volatile float *temp2 = temp;
            temp2[ Idx ] += temp2[ Idx + 32 ];
            temp2[ Idx ] += temp2[ Idx + 16 ];
            temp2[ Idx ] += temp2[ Idx + 8 ];
            temp2[ Idx ] += temp2[ Idx + 4 ];
            temp2[ Idx ] += temp2[ Idx + 2 ];
            temp2[ Idx ] += temp2[ Idx + 1 ];
        }
    #endif

    if ( Idx == 0 ){
            vtmp[ blockIdx.x ] = temp[ 0 ];
    }
}


// SELLP SpMV kernel
// see paper by M. KREUTZER, G. HAGER, G WELLEIN, H. FEHSKE A. BISHOP
// A UNIFIED SPARSE MATRIX DATA FORMAT 
// FOR MODERN PROCESSORS WITH WIDE SIMD UNITS
// SELLC SpMV kernel modified assigning multiple threads to each row - 2D kernel
__global__ void 
magma_zcgmerge_spmvsellpt_kernel_8( int num_rows, 
                     int blocksize,
                     int T,
                     magmaDoubleComplex *d_val, 
                     magma_index_t *d_colind,
                     magma_index_t *d_rowptr,
                     magmaDoubleComplex *d,
                     magmaDoubleComplex *z)
{
   // T threads assigned to each row
    int idx = threadIdx.y ;     // thread in row
    int idy = threadIdx.x;      // local row
    int ldx = idx * blocksize + idy;
    int bdx = blockIdx.y * gridDim.x + blockIdx.x; // global block index
    int row = bdx * blocksize + idy;  // global row index

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
            dot += val * d[ col ];
        }
        shared[ldx]  = dot;

        __syncthreads();
        if( idx < 4 ){
            shared[ldx]+=shared[ldx+blocksize*4];              
            __syncthreads();
            if( idx < 2 ) shared[ldx]+=shared[ldx+blocksize*2];   
            __syncthreads();
            if( idx == 0 ) {
                z[row] = 
                (shared[ldx]+shared[ldx+blocksize*1]);
            }

        }

    }
}
// SELLP SpMV kernel
// see paper by M. KREUTZER, G. HAGER, G WELLEIN, H. FEHSKE A. BISHOP
// A UNIFIED SPARSE MATRIX DATA FORMAT 
// FOR MODERN PROCESSORS WITH WIDE SIMD UNITS
// SELLC SpMV kernel modified assigning multiple threads to each row - 2D kernel
__global__ void 
magma_zcgmerge_spmvsellpt_kernel_16( int num_rows, 
                     int blocksize,
                     int T,
                     magmaDoubleComplex *d_val, 
                     magma_index_t *d_colind,
                     magma_index_t *d_rowptr,
                     magmaDoubleComplex *d,
                     magmaDoubleComplex *z)
{
   // T threads assigned to each row
    int idx = threadIdx.y ;     // thread in row
    int idy = threadIdx.x;      // local row
    int ldx = idx * blocksize + idy;
    int bdx = blockIdx.y * gridDim.x + blockIdx.x; // global block index
    int row = bdx * blocksize + idy;  // global row index

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
            dot += val * d[ col ];
        }
        shared[ldx]  = dot;

        __syncthreads();
        if( idx < 8 ){
            shared[ldx]+=shared[ldx+blocksize*8];              
            __syncthreads();
            if( idx < 4 ) shared[ldx]+=shared[ldx+blocksize*4];   
            __syncthreads();
            if( idx < 2 ) shared[ldx]+=shared[ldx+blocksize*2];   
            __syncthreads();
            if( idx == 0 ) {
                z[row] = 
                (shared[ldx]+shared[ldx+blocksize*1]);
            }

        }

    }
}


// SELLP SpMV kernel
// see paper by M. KREUTZER, G. HAGER, G WELLEIN, H. FEHSKE A. BISHOP
// A UNIFIED SPARSE MATRIX DATA FORMAT 
// FOR MODERN PROCESSORS WITH WIDE SIMD UNITS
// SELLC SpMV kernel modified assigning multiple threads to each row - 2D kernel
__global__ void 
magma_zcgmerge_spmvsellpt_kernel_32( int num_rows, 
                     int blocksize,
                     int T,
                     magmaDoubleComplex *d_val, 
                     magma_index_t *d_colind,
                     magma_index_t *d_rowptr,
                     magmaDoubleComplex *d,
                     magmaDoubleComplex *z)
{
   // T threads assigned to each row
    int idx = threadIdx.y ;     // thread in row
    int idy = threadIdx.x;      // local row
    int ldx = idx * blocksize + idy;
    int bdx = blockIdx.y * gridDim.x + blockIdx.x; // global block index
    int row = bdx * blocksize + idy;  // global row index

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
            dot += val * d[ col ];
        }
        shared[ldx]  = dot;

        __syncthreads();
        if( idx < 16 ){
            shared[ldx]+=shared[ldx+blocksize*16];              
            __syncthreads();
            if( idx < 8 ) shared[ldx]+=shared[ldx+blocksize*8];  
            __syncthreads();
            if( idx < 4 ) shared[ldx]+=shared[ldx+blocksize*4];   
            __syncthreads();
            if( idx < 2 ) shared[ldx]+=shared[ldx+blocksize*2];   
            __syncthreads();
            if( idx == 0 ) {
                z[row] = 
                (shared[ldx]+shared[ldx+blocksize*1]);
            }

        }

    }
}


// kernel to handle scalars
__global__ void // rho = beta/tmp; gamma = beta;
magma_zcg_rhokernel(  
                    magmaDoubleComplex *skp ){
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if( i==0 ){
        magmaDoubleComplex tmp = skp[1];
        skp[3] = tmp/skp[4];
        skp[2] = tmp;
    }
}

/**
    Purpose
    -------

    Merges the first SpmV using different formats with the dot product 
    and the computation of rho

    Arguments
    ---------

    @param
    A           magma_z_sparse_matrix
                input matrix 

    @param
    d1          magmaDoubleComplex*
                temporary vector

    @param
    d2          magmaDoubleComplex*
                temporary vector

    @param
    d_d         magmaDoubleComplex*
                input vector d

    @param
    d_z         magmaDoubleComplex*
                input vector z

    @param
    skp         magmaDoubleComplex*
                array for parameters ( skp[3]=rho )


    @ingroup magmasparse_zgegpuk
    ********************************************************************/

extern "C" magma_int_t
magma_zcgmerge_spmv1(  
                 magma_z_sparse_matrix A,
                 magmaDoubleComplex *d1,
                 magmaDoubleComplex *d2,
                 magmaDoubleComplex *d_d,
                 magmaDoubleComplex *d_z,
                 magmaDoubleComplex *skp ){

    int local_block_size=256;
    dim3 Bs( local_block_size );
    dim3 Gs( (A.num_rows+local_block_size-1)/local_block_size );
    dim3 Gs_next;
    int Ms =  local_block_size * sizeof( magmaDoubleComplex ); 
    magmaDoubleComplex *aux1 = d1, *aux2 = d2;
    int b = 1;        

    if( A.storage_type == Magma_CSR )
        magma_zcgmerge_spmvcsr_kernel<<<Gs, Bs, Ms, magma_stream >>>
        ( A.num_rows, A.val, A.row, A.col, d_d, d_z, d1 );
    else if( A.storage_type == Magma_ELLPACK )
        magma_zcgmerge_spmvellpack_kernel<<<Gs, Bs, Ms, magma_stream >>>
        ( A.num_rows, A.max_nnz_row, A.val, A.col, d_d, d_z, d1 );
    else if( A.storage_type == Magma_ELL )
        magma_zcgmerge_spmvellpackt_kernel<<<Gs, Bs, Ms, magma_stream >>>
        ( A.num_rows, A.max_nnz_row, A.val, A.col, d_d, d_z, d1 );
    else if( A.storage_type == Magma_SELLC || A.storage_type == Magma_SELLP ){
        if( A.blocksize==256){
            magma_zcgmerge_spmvsellc_kernel<<<Gs, Bs, Ms, magma_stream >>>
            ( A.num_rows, A.blocksize, A. val, A.col, A.row,  
                d_d, d_z, d1 );
        }
        else
            printf("error: SELLC only for blocksize 256.\n");
    }
    else if( A.storage_type == Magma_SELLP ){
            int num_threadssellp = A.blocksize*A.alignment;
            magma_int_t arch = magma_getdevice_arch();
            if ( arch < 200 && num_threadssellp > 256 )
                printf("error: too much shared memory requested.\n");

            dim3 block( A.blocksize, A.alignment, 1);
            int dimgrid1 = sqrt(A.numblocks);
            int dimgrid2 = (A.numblocks + dimgrid1 -1 ) / dimgrid1;

            dim3 gridsellp( dimgrid1, dimgrid2, 1);
            int Mssellp = num_threadssellp * sizeof( magmaDoubleComplex );

            if( A.alignment == 8)
                magma_zcgmerge_spmvsellpt_kernel_8
                <<< gridsellp, block, Mssellp, magma_stream >>>
                ( A.num_rows, A.blocksize, A.alignment, 
                    A.val, A.col, A.row, d_d, d_z);

            else if( A.alignment == 16)
                magma_zcgmerge_spmvsellpt_kernel_16
                <<< gridsellp, block, Mssellp, magma_stream >>>
                ( A.num_rows, A.blocksize, A.alignment, 
                    A.val, A.col, A.row, d_d, d_z);

            else if( A.alignment == 32)
                magma_zcgmerge_spmvsellpt_kernel_32
                <<< gridsellp, block, Mssellp, magma_stream >>>
                ( A.num_rows, A.blocksize, A.alignment, 
                    A.val, A.col, A.row, d_d, d_z);

            else
                printf("error: alignment not supported.\n");

        // in case of using SELLP, we can't efficiently merge the 
        // dot product and the first reduction loop into the SpMV kernel
        // as the SpMV grid would result in low occupancy.
        magma_zcgmerge_spmvellpackrt_kernel2<<<Gs, Bs, Ms, magma_stream >>>
                              ( A.num_rows, d_z, d_d, d1 );

    }
    else if( A.storage_type == Magma_ELLRT ){
        // in case of using ELLRT, we need a different grid, assigning
        // threads_per_row processors to each row
        // the block size is num_threads
        // fixed values


    int num_blocks = ( (A.num_rows+A.blocksize-1)/A.blocksize);

    int num_threads = A.alignment*A.blocksize;

    int real_row_length = ((int)(A.max_nnz_row+A.alignment-1)/A.alignment)
                            *A.alignment;

    magma_int_t arch = magma_getdevice_arch();
    if ( arch < 200 && num_threads > 256 )
        printf("error: too much shared memory requested.\n");

    int dimgrid1 = sqrt(num_blocks);
    int dimgrid2 = (num_blocks + dimgrid1 -1 ) / dimgrid1;
    dim3 gridellrt( dimgrid1, dimgrid2, 1);

    int Mellrt = A.alignment * A.blocksize * sizeof( magmaDoubleComplex );
    // printf("launch kernel: %dx%d %d %d\n", grid.x, grid.y, num_threads , Ms);

    if( A.alignment == 32 ){
        magma_zcgmerge_spmvellpackrt_kernel_32
                <<< gridellrt, num_threads , Mellrt, magma_stream >>>
                 ( A.num_rows, A.val, A.col, A.row, d_d, d_z, d1, 
                                                 A.alignment, real_row_length );
    }
    else if( A.alignment == 16 ){
        magma_zcgmerge_spmvellpackrt_kernel_16
                <<< gridellrt, num_threads , Mellrt, magma_stream >>>
                 ( A.num_rows, A.val, A.col, A.row, d_d, d_z, d1, 
                                                 A.alignment, real_row_length );
    }
    else if( A.alignment == 8 ){
        magma_zcgmerge_spmvellpackrt_kernel_8
                <<< gridellrt, num_threads , Mellrt, magma_stream >>>
                 ( A.num_rows, A.val, A.col, A.row, d_d, d_z, d1, 
                                                 A.alignment, real_row_length );
    }
    else{
        printf("error: alignment %d not supported.\n", A.alignment);
        exit(-1);
    }
        // in case of using ELLRT, we can't efficiently merge the 
        // dot product and the first reduction loop into the SpMV kernel
        // as the SpMV grid would result in low occupancy.

        magma_zcgmerge_spmvellpackrt_kernel2<<<Gs, Bs, Ms, magma_stream >>>
                              ( A.num_rows, d_z, d_d, d1 );
    }

    while( Gs.x > 1 ){
        Gs_next.x = ( Gs.x+Bs.x-1 )/ Bs.x ;
        if( Gs_next.x == 1 ) Gs_next.x = 2;
        magma_zcgreduce_kernel_spmv1<<< Gs_next.x/2, Bs.x/2, Ms/2 >>> 
                                        ( Gs.x,  A.num_rows, aux1, aux2 );
        Gs_next.x = Gs_next.x /2;
        Gs.x = Gs_next.x;
        b = 1 - b;
        if( b ){ aux1 = d1; aux2 = d2; }
        else   { aux2 = d1; aux1 = d2; }
    }


    magma_zcopyvector( 1, aux1, 1, skp+4, 1 );
    dim3 Bs2( 2 );
    dim3 Gs2( 1 );
    magma_zcg_rhokernel<<<Gs2, Bs2, 0>>>( skp );

   return MAGMA_SUCCESS;
}


/* -------------------------------------------------------------------------- */

// updates x and r and computes the first part of the dot product r*r
__global__ void 
magma_zcgmerge_xrbeta_kernel(  
                    int n, 
                    magmaDoubleComplex *x, 
                    magmaDoubleComplex *r,
                    magmaDoubleComplex *d,
                    magmaDoubleComplex *z,
                    magmaDoubleComplex *skp,
                    magmaDoubleComplex *vtmp
                                            ){

    extern __shared__ magmaDoubleComplex temp[]; 
    int Idx = threadIdx.x;   
    int i   = blockIdx.x * blockDim.x + Idx;

    magmaDoubleComplex rho = skp[3];
    magmaDoubleComplex mrho = MAGMA_Z_MAKE( -1.0, 0.0)*rho;

    temp[ Idx ] = MAGMA_Z_MAKE( 0.0, 0.0);

    if( i<n ){
        x[i] += rho * d[i] ;
        r[i] += mrho  * z[i];
        temp[ Idx ] = r[i] * r[i];
    }
    __syncthreads();
    if ( Idx < 128 ){
        temp[ Idx ] += temp[ Idx + 128 ];
    }
    __syncthreads();
    if ( Idx < 64 ){
        temp[ Idx ] += temp[ Idx + 64 ];
    }
    __syncthreads();
    #if defined(PRECISION_z) || defined(PRECISION_c)
        if( Idx < 32 ){
            temp[ Idx ] += temp[ Idx + 32 ];__syncthreads();
            temp[ Idx ] += temp[ Idx + 16 ];__syncthreads();
            temp[ Idx ] += temp[ Idx + 8 ];__syncthreads();
            temp[ Idx ] += temp[ Idx + 4 ];__syncthreads();
            temp[ Idx ] += temp[ Idx + 2 ];__syncthreads();
            temp[ Idx ] += temp[ Idx + 1 ];__syncthreads();
        }
    #endif
    #if defined(PRECISION_d)
        if( Idx < 32 ){
            volatile double *temp2 = temp;
            temp2[ Idx ] += temp2[ Idx + 32 ];
            temp2[ Idx ] += temp2[ Idx + 16 ];
            temp2[ Idx ] += temp2[ Idx + 8 ];
            temp2[ Idx ] += temp2[ Idx + 4 ];
            temp2[ Idx ] += temp2[ Idx + 2 ];
            temp2[ Idx ] += temp2[ Idx + 1 ];
        }
    #endif
    #if defined(PRECISION_s)
        if( Idx < 32 ){
            volatile float *temp2 = temp;
            temp2[ Idx ] += temp2[ Idx + 32 ];
            temp2[ Idx ] += temp2[ Idx + 16 ];
            temp2[ Idx ] += temp2[ Idx + 8 ];
            temp2[ Idx ] += temp2[ Idx + 4 ];
            temp2[ Idx ] += temp2[ Idx + 2 ];
            temp2[ Idx ] += temp2[ Idx + 1 ];
        }
    #endif

    if ( Idx == 0 ){
            vtmp[ blockIdx.x ] = temp[ 0 ];
    }

}

// kernel to handle scalars
__global__ void //alpha = beta / gamma
magma_zcg_alphabetakernel(  
                    magmaDoubleComplex *skp ){
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if( i==0 ){
        magmaDoubleComplex tmp1 = skp[1];
        skp[0] =  tmp1/skp[2];
        //printf("beta=%e\n", MAGMA_Z_REAL(tmp1));
    }
}

// update search Krylov vector d
__global__ void 
magma_zcg_d_kernel(  
                    int n, 
                    magmaDoubleComplex *skp,
                    magmaDoubleComplex *r,
                    magmaDoubleComplex *d
                                           ){
  
    int i   = blockIdx.x * blockDim.x + threadIdx.x;

    magmaDoubleComplex alpha = skp[0];

    if( i<n ){
        d[i] = r[i] + alpha * d[i];
    }

}



/**
    Purpose
    -------

    Merges the update of r and x with the dot product and performs then 
    the update for the Krylov vector d

    Arguments
    ---------

    @param
    n           int
                dimension n

    @param
    d1          magmaDoubleComplex*
                temporary vector

    @param
    d2          magmaDoubleComplex*
                temporary vector

    @param
    d_x         magmaDoubleComplex*
                input vector x

    @param
    d_r         magmaDoubleComplex*
                input/output vector r

    @param
    d_d         magmaDoubleComplex*
                input vector d

    @param
    d_z         magmaDoubleComplex*
                input vector z
    @param
    skp         magmaDoubleComplex*
                array for parameters


    @ingroup magmasparse_zsygpuk
    ********************************************************************/

extern "C" magma_int_t
magma_zcgmerge_xrbeta(  
                 int n,
                 magmaDoubleComplex *d1,
                 magmaDoubleComplex *d2,
                 magmaDoubleComplex *d_x,
                 magmaDoubleComplex *d_r,
                 magmaDoubleComplex *d_d,
                 magmaDoubleComplex *d_z, 
                 magmaDoubleComplex *skp ){

    int local_block_size=256;
    dim3 Bs( local_block_size );
    dim3 Gs( (n+local_block_size-1)/local_block_size );
    dim3 Gs_next;
    int Ms =  2*local_block_size * sizeof( magmaDoubleComplex ); 
    magmaDoubleComplex *aux1 = d1, *aux2 = d2;
    int b = 1;        
    magma_zcgmerge_xrbeta_kernel<<<Gs, Bs, Ms>>>
                                    ( n, d_x, d_r, d_d, d_z, skp, d1);  



    while( Gs.x > 1 ){
        Gs_next.x = ( Gs.x+Bs.x-1 )/ Bs.x ;
        if( Gs_next.x == 1 ) Gs_next.x = 2;
        magma_zcgreduce_kernel_spmv1<<< Gs_next.x/2, Bs.x/2, Ms/2 >>> 
                                    ( Gs.x, n, aux1, aux2 );
        Gs_next.x = Gs_next.x /2;
        Gs.x = Gs_next.x;
        b = 1 - b;
        if( b ){ aux1 = d1; aux2 = d2; }
        else   { aux2 = d1; aux1 = d2; }
    }


    magma_zcopyvector( 1, aux1, 1, skp+1, 1 );
    dim3 Bs2( 2 );
    dim3 Gs2( 1 );
    magma_zcg_alphabetakernel<<<Gs2, Bs2, 0>>>( skp );

    dim3 Bs3( local_block_size );
    dim3 Gs3( (n+local_block_size-1)/local_block_size );
    magma_zcg_d_kernel<<<Gs3, Bs3, 0>>>( n, skp, d_r, d_d );  

   return MAGMA_SUCCESS;
}

/* -------------------------------------------------------------------------- */

