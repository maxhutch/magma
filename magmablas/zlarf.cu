/*
    -- MAGMA (version 1.5.0-beta3) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date July 2014

       @precisions normal z -> s d c

*/
#include "common_magma.h"
#include "magma_templates.h"

// 512 is maximum number of threads for CUDA capability 1.x
#define BLOCK_SIZE 512

#define BLOCK_SIZEx  32
#define BLOCK_SIZEy  16


//==============================================================================

__global__
void magma_zlarf_kernel( int m, magmaDoubleComplex *v, magmaDoubleComplex *tau,
                         magmaDoubleComplex *c, int ldc, double *xnorm )
{
    if ( !MAGMA_Z_EQUAL(*tau, MAGMA_Z_ZERO) ) {
        const int i = threadIdx.x;
        magmaDoubleComplex *dc = c + blockIdx.x * ldc;

        __shared__ magmaDoubleComplex sum[ BLOCK_SIZE ];
        magmaDoubleComplex lsum;

        /*  w := v' * C  */
        lsum = MAGMA_Z_ZERO;
        for( int j = i; j < m; j += BLOCK_SIZE ){
            if (j==0)
               lsum += MAGMA_Z_MUL( MAGMA_Z_ONE, dc[j] );
            else
               lsum += MAGMA_Z_MUL( MAGMA_Z_CNJG( v[j] ), dc[j] );
        }
        sum[i] = lsum;
        magma_sum_reduce< BLOCK_SIZE >( i, sum );

        /*  C := C - v * w  */
        __syncthreads();
        magmaDoubleComplex z__1 = - MAGMA_Z_CNJG(*tau) * sum[0];
        for( int j = m-i-1; j>=0 ; j -= BLOCK_SIZE ) {
             if (j==0)
                dc[j] += z__1;
             else
                dc[j] += z__1 * v[j];
        }
        __syncthreads();

        /* Adjust the rest of the column norms */
        //if (i==0){
        //    double temp = MAGMA_Z_ABS( dc[0] ) / xnorm[blockIdx.x];
        //    temp = (temp + 1.) * (1. - temp);
        //    xnorm[blockIdx.x] = xnorm[blockIdx.x] * sqrt(temp); 
        //}
    }
}

//==============================================================================

__global__
void magma_zlarf_smkernel( int m, int n, magmaDoubleComplex *v, magmaDoubleComplex *tau,
                           magmaDoubleComplex *c, int ldc, double *xnorm )
{
    if ( ! MAGMA_Z_EQUAL(*tau, MAGMA_Z_ZERO) ) {
        const int i = threadIdx.x, col= threadIdx.y;

        for( int k = col; k < n; k += BLOCK_SIZEy ) {
            magmaDoubleComplex *dc = c + k * ldc;
    
            __shared__ magmaDoubleComplex sum[ BLOCK_SIZEx ][ BLOCK_SIZEy + 1];
            magmaDoubleComplex lsum;
    
            /*  w := v' * C  */
            lsum = MAGMA_Z_ZERO;
            for( int j = i; j < m; j += BLOCK_SIZEx ){
                if (j==0)
                   lsum += MAGMA_Z_MUL( MAGMA_Z_ONE, dc[j] );
                else
                   lsum += MAGMA_Z_MUL( MAGMA_Z_CNJG( v[j] ), dc[j] );
            }
            sum[i][col] = lsum;
            magma_sum_reduce_2d< BLOCK_SIZEx, BLOCK_SIZEy+1 >( i, col, sum );
    
            /*  C := C - v * w  */
            __syncthreads();
            magmaDoubleComplex z__1 = - MAGMA_Z_CNJG(*tau) * sum[0][col];
            for( int j = m-i-1; j>=0 ; j -= BLOCK_SIZEx ) {
                 if (j==0)
                    dc[j] += z__1;
                 else
                    dc[j] += z__1 * v[j];
            }
            __syncthreads();
    
            /* Adjust the rest of the column norms */
            // if (i==0){
            //    double temp = MAGMA_Z_ABS( dc[0] ) / xnorm[k];
            //    temp = (temp + 1.) * (1. - temp);
            //    xnorm[k] = xnorm[k] * sqrt(temp);
            // }
        }
    }
}

//==============================================================================

/*
    Apply a complex elementary reflector H to a complex M-by-N
    matrix C from the left. H is represented in the form
          H = I - tau * v * v'
    where tau is a complex scalar and v is a complex vector.
    If tau = 0, then H is taken to be the unit matrix.

    To apply H' (the conjugate transpose of H), supply conjg(tau)
    instead tau.

    This routine uses only one SM (block).
 */
extern "C" void
magma_zlarf_sm(int m, int n, magmaDoubleComplex *v, magmaDoubleComplex *tau,
               magmaDoubleComplex *c, int ldc, double *xnorm)
{
    dim3  blocks( 1 );
    dim3 threads( BLOCK_SIZEx, BLOCK_SIZEy );

    magma_zlarf_smkernel<<< blocks, threads, 0, magma_stream >>>( m, n, v, tau, c, ldc, xnorm);
}

//==============================================================================
/*
    Apply a complex elementary reflector H to a complex M-by-N
    matrix C from the left. H is represented in the form
          H = I - tau * v * v'
    where tau is a complex scalar and v is a complex vector.
    If tau = 0, then H is taken to be the unit matrix.

    To apply H' (the conjugate transpose of H), supply conjg(tau) 
    instead tau.

    The norms of v(:, 1:n) are given as input in xnorm(1:n). On exit, the norms
    are adjusted to hold the norms of v(2:m,2:n). This is a difference with the 
    LAPACK's zlarf routine. 
 */

extern "C" magma_int_t
magma_zlarf_gpu(
    magma_int_t m,  magma_int_t n,
    magmaDoubleComplex *v, magmaDoubleComplex *tau,
    magmaDoubleComplex *c,  magma_int_t ldc, double *xnorm)
{
    dim3  blocks( n );
    dim3 threads( BLOCK_SIZE );

    magma_zlarf_kernel<<< blocks, threads, 0, magma_stream >>>( m, v, tau, c, ldc, xnorm);

    // The computation can be done on 1 SM with the following routine.
    // magma_zlarf_sm(m, n, v, tau, c, ldc, xnorm);

    return MAGMA_SUCCESS;
}

//==============================================================================
