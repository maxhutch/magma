/*
    -- MAGMA (version 1.6.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date January 2015

       @generated from zlarf.cu normal z -> s, Fri Jan 30 19:00:08 2015
       @author Azzam Haidar

*/
#include "common_magma.h"
#include "magma_templates.h"

// 512 is maximum number of threads for CUDA capability 1.x
#define BLOCK_SIZE 512

#define BLOCK_SIZEx  32
#define BLOCK_SIZEy  16


//==============================================================================
//==============================================================================

__global__
void magma_slarf_kernel( int m, const float *dv, const float *dtau,
                         float *dc, int lddc )
{
    if ( !MAGMA_S_EQUAL(*dtau, MAGMA_S_ZERO) ) {
        const int tx = threadIdx.x;
        dc = dc + blockIdx.x * lddc;

        __shared__ float sum[ BLOCK_SIZE ];
        float tmp;

        /* perform  w := v**H * C  */
        if (tx==0)
            tmp = dc[0]; //since V[0] should be one
        else
            tmp = MAGMA_S_ZERO;
        for( int j = tx+1; j < m; j += BLOCK_SIZE ){
            tmp += MAGMA_S_MUL( MAGMA_S_CNJG( dv[j] ), dc[j] );
        }
        sum[tx] = tmp;
        magma_sum_reduce< BLOCK_SIZE >( tx, sum );

        /*  C := C - v * w  */
        __syncthreads();
        tmp = - MAGMA_S_CNJG(*dtau) * sum[0];
        for( int j = m-tx-1; j>0 ; j -= BLOCK_SIZE )
             dc[j] += tmp * dv[j];

        if(tx==0) dc[0] += tmp;
    }
}

//==============================================================================
//==============================================================================

__global__
void magma_slarf_smkernel( int m, int n, float *dv, float *dtau,
                           float *dc, int lddc )
{
    if ( ! MAGMA_S_EQUAL(*dtau, MAGMA_S_ZERO) ) {
        const int i = threadIdx.x, col= threadIdx.y;

        for( int k = col; k < n; k += BLOCK_SIZEy ) {
            dc = dc + k * lddc;
    
            __shared__ float sum[ BLOCK_SIZEx ][ BLOCK_SIZEy + 1];
            float lsum;
    
            /*  w := v**H * C  */
            lsum = MAGMA_S_ZERO;
            for( int j = i; j < m; j += BLOCK_SIZEx ){
                if (j==0)
                   lsum += MAGMA_S_MUL( MAGMA_S_ONE, dc[j] );
                else
                   lsum += MAGMA_S_MUL( MAGMA_S_CNJG( dv[j] ), dc[j] );
            }
            sum[i][col] = lsum;
            magma_sum_reduce_2d< BLOCK_SIZEx, BLOCK_SIZEy+1 >( i, col, sum );
    
            /*  C := C - v * w  */
            __syncthreads();
            float z__1 = - MAGMA_S_CNJG(*dtau) * sum[0][col];
            for( int j = m-i-1; j>=0 ; j -= BLOCK_SIZEx ) {
                 if (j==0)
                    dc[j] += z__1;
                 else
                    dc[j] += z__1 * dv[j];
            }
        }
    }
}

//==============================================================================

/*
    Apply a real elementary reflector H to a real M-by-N
    matrix C from the left. H is represented in the form
          H = I - tau * v * v**H
    where tau is a real scalar and v is a real vector.
    If tau = 0, then H is taken to be the unit matrix.

    To apply H**H (the conjugate transpose of H), supply conjg(tau)
    instead tau.

    This routine uses only one SM (block).
 */
extern "C" void
magma_slarf_sm(magma_int_t m, magma_int_t n, float *dv, float *dtau,
               float *dc, magma_int_t lddc)
{
    dim3  blocks( 1 );
    dim3 threads( BLOCK_SIZEx, BLOCK_SIZEy );

    magma_slarf_smkernel<<< blocks, threads, 0, magma_stream >>>( m, n, dv, dtau, dc, lddc );
}
//==============================================================================
/*
    Apply a real elementary reflector H to a real M-by-N
    matrix C from the left. H is represented in the form
          H = I - tau * v * v**H
    where tau is a real scalar and v is a real vector.
    If tau = 0, then H is taken to be the unit matrix.

    To apply H**H (the conjugate transpose of H), supply conjg(tau) 
    instead tau.

 */

extern "C" magma_int_t
magma_slarf_gpu(
    magma_int_t m,  magma_int_t n,
    magmaFloat_const_ptr dv,
    magmaFloat_const_ptr dtau,
    magmaFloat_ptr dC,  magma_int_t lddc)
{
    dim3 grid( n, 1, 1 );
    dim3 threads( BLOCK_SIZE );
    if ( n > 0 ) {
        magma_slarf_kernel<<< grid, threads, 0, magma_stream >>>( m, dv, dtau, dC, lddc);
    }

    // The computation can be done on 1 SM with the following routine.
    // magma_slarf_sm(m, n, dv, dtau, dc, lddc);

    return MAGMA_SUCCESS;
}

//==============================================================================
