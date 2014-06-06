/*
    -- MAGMA (version 1.4.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       December 2013

       @generated s Tue Dec 17 13:18:45 2013

*/
#include "common_magma.h"

//#if (GPUSHMEM < 200)
   #define BLOCK_SIZE 512
//#else
//   #define BLOCK_SIZE 768
//#endif

__global__ void 
magma_strmv_kernel2(const float *T, int ldt,
                    float *v, float *y, float *tau);

__global__ void 
magma_sgemv_kernel3(int m, const float * __restrict__ V, int ldv,
                    float *c, float *dwork,
                    float *tau);


//////////////////////////////////////////////////////////////////////////////

__global__ void
magma_sgemv_kernel1(int m, const float * __restrict__ V, int ldv,
                    const float * __restrict__ c,
                    float *dwork);
__global__ void
magma_sgemv_kernel2(int m, int n, const float * __restrict__ V, int ldv,
                    const float * __restrict__ x, float *c);
__global__ void 
magma_strmv_tkernel(float *T, int ldt, float *v,
                                    float *y);
__global__ void
magma_snrm2_adjust_kernel(float *xnorm, float *c);

extern "C" magma_int_t
magma_sgeqr2x4_gpu(magma_int_t *m, magma_int_t *n, float *dA, 
                   magma_int_t *ldda, float *dtau,
                   float *dT, float *ddA,
                   float *dwork, magma_int_t *info, magma_queue_t stream)
{
/*  -- MAGMA (version 1.4.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       December 2013

    Purpose   
    =======   
    SGEQR2 computes a QR factorization of a real m by n matrix A:   
    A = Q * R.

    This expert routine requires two more arguments than the standard 
    sgeqr2, namely, dT and ddA, explained below. The storage for A is 
    also not as in the LAPACK's sgeqr2 routine (see below). 

    The first is used to output the triangular 
    n x n factor T of the block reflector used in the factorization. 
    The second holds the diagonal nxn blocks of A, i.e., the diagonal
    submatrices of R. This routine implements the left looking QR.

    This version adds internal blocking.

    Arguments   
    =========   
    M       (input) INTEGER   
            The number of rows of the matrix A.  M >= 0.   

    N       (input) INTEGER   
            The number of columns of the matrix A.  N >= 0.   

    A       (input/output) REAL array, dimension (LDA,N)   
            On entry, the m by n matrix A.   
            On exit, the unitary matrix Q as a
            product of elementary reflectors (see Further Details).

            the elements on and above the diagonal of the array   
            contain the min(m,n) by n upper trapezoidal matrix R (R is   
            upper triangular if m >= n); the elements below the diagonal,   
            with the array TAU, represent the unitary matrix Q as a   
            product of elementary reflectors (see Further Details).   

    LDA     (input) INTEGER   
            The leading dimension of the array A.  LDA >= max(1,M).   

    TAU     (output) REAL array, dimension (min(M,N))   
            The scalar factors of the elementary reflectors (see Further   
            Details).   

    dT      (output) REAL array, dimension N x N.
            Stores the triangular N x N factor T of the block reflector 
            used in the factorization. The lower triangular part is 0.

    ddA     (output) REAL array, dimension N x N.
            Stores the elements of the upper N x N diagonal block of A.
            LAPACK stores this array in A. There are 0s below the diagonal.

    RWORK   (workspace) DOUBLE_PRECISION array, dimension (3 N)

    INFO    (output) INTEGER   
            = 0: successful exit   
            < 0: if INFO = -i, the i-th argument had an illegal value   

    Further Details   
    ===============   
    The matrix Q is represented as a product of elementary reflectors   

       Q = H(1) H(2) . . . H(k), where k = min(m,n).   

    Each H(i) has the form   

       H(i) = I - tau * v * v'   

    where tau is a real scalar, and v is a real vector with   
    v(1:i-1) = 0 and v(i) = 1; v(i+1:m) is stored on exit in A(i+1:m,i),   
    and tau in TAU(i).   
    =====================================================================    */

    #define da_ref(a_1,a_2) ( dA+(a_2)*(*ldda) + (a_1))
    #define dt_ref(a_1,a_2) ( dT+(a_2)*(k) + (a_1))
    #define BS 32

    magma_int_t i, k;

    float *dnorm = (float *)dwork;
    float *work = (float *)(dwork+2*(*n));

    magma_queue_t cstream;
    magmablasGetKernelStream(&cstream);
    magmablasSetKernelStream(stream);

    *info = 0;
    if (*m < 0) {
        *info = -1;
    } else if (*n < 0) {
        *info = -2;
    } else if (*ldda < max(1,*m)) {
        *info = -4;
    }
    if (*info != 0) {
        magma_xerbla( __func__, -(*info) );
        return *info;
    }

    /* Compute the norms of the trailing columns */
    k = min(*m,*n);
    magmablas_snrm2_cols(*m, k, da_ref(0,0), *ldda, dnorm);

    for (magma_int_t b=0; b < k; b += BS) {
        for (i = b; i < min(k, b+BS); ++i) {

            /*   Apply H' to A(:,i) from the left                           */    
            if ( i-b > 0){
                magma_sgemv_kernel3<<< i-1, BLOCK_SIZE, 0, magma_stream >>>( *m-i+1, da_ref(i-1,0), *ldda,
                                                    da_ref(i-1, i-1), work, dtau+i-1);
                magma_strmv_kernel2<<< i-1, i-1, 0, magma_stream >>>( dt_ref(0,0), k, work,
                                                    dt_ref(0,i-1), dtau+i-1);

                /* dwork = V' c                   */
                magma_sgemv_kernel1<<< i-b, BLOCK_SIZE, 0, magma_stream >>>(*m-b, da_ref(b, b), 
                             *ldda, da_ref(b,i), work);

                /* dwork = T' work                */
                magma_strmv_tkernel<<< i-b, i-b, 0, magma_stream >>>(dt_ref(b,b), k, work, work+i-b);

                /* c = c - V work                 */
                dim3  blocks3( (*m-b + BLOCK_SIZE-1) / BLOCK_SIZE );
                dim3 threads3( BLOCK_SIZE );
                magma_sgemv_kernel2<<< blocks3, threads3, 0, magma_stream >>>(*m-b, i-b, da_ref(b,b), *ldda, 
                                   work+i-b, da_ref(b, i));
            }

            /*   Adjust the dnorm[i] to hold the norm of A(i:m,i)           */ 
            if ( i > 0 )
                magma_snrm2_adjust_kernel<<< 1, i, 0, magma_stream >>> (dnorm+i, da_ref(0, i));
            
            /*  Generate elementary reflector H(i) to annihilate A(i+1:m,i) 
                1. 1 is not yet put on the diagonal of A
                2. Elements above the diagonal are copied in ddA and
                   the ones in A are set to zero                                         
                3. update T                                                 */
            magma_slarfgx_gpu(*m-i, da_ref(i, i), da_ref(min(i+1,*m),i), dtau+i, 
                              dnorm+i, ddA + i + i*(*n), i);

            if (i==0){
              float tt = MAGMA_S_ONE;
              magmablas_slacpy(MagmaUpperLower, 1, 1, dtau, 1, dt_ref(0,0), 1);
              magma_ssetmatrix(1,1, &tt,1, da_ref(i, i),1);
            }
/*
            else
             {
                // Compute the i-th column of T.
                //   Set da_ref(i, i) = 1.                                    
                magma_sgemv_kernel3<<< i, BLOCK_SIZE, 0, magma_stream >>>( *m-i, da_ref(i,0), *ldda, 
                                          da_ref(i, i), work, dtau+i);
                magma_strmv_kernel2<<< i, i, 0, magma_stream          >>>( dt_ref(0,0), k, work, 
                                                          dt_ref(0,i), dtau+i);
              }
*/

        }
        magma_sgemv_kernel3<<< i-1, BLOCK_SIZE, 0, magma_stream >>>( *m-i+1, da_ref(i-1,0), *ldda,
                                                    da_ref(i-1, i-1), work, dtau+i-1);
        magma_strmv_kernel2<<< i-1, i-1, 0, magma_stream >>>( dt_ref(0,0), k, work,
                                                    dt_ref(0,i-1), dtau+i-1);

        
        /* Apply the transformations to the trailing matrix. */
        //magma_slarfb2_gpu( MagmaLeft, MagmaTrans, MagmaForward, MagmaColumnwise,
        magma_slarfb2_gpu(
                           *m-b, k-i, BS,
                           da_ref(b, b), *ldda, dT+b+b*k, k,
                           da_ref(b, i), *ldda, work, k-i);
    }

    magmablasSetKernelStream(cstream);

    return *info;
} /* magma_sgeqr2 */
