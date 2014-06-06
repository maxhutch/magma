/*
    -- MAGMA (version 1.4.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       December 2013

       @generated d Tue Dec 17 13:18:36 2013

*/
#include <math.h>
#include "common_magma.h"


extern "C" magma_int_t
magma_dgetrf_mgpu(magma_int_t num_gpus,
                 magma_int_t m, magma_int_t n,
                 double **d_lA, magma_int_t ldda,
                 magma_int_t *ipiv, magma_int_t *info)
{
/*  -- MAGMA (version 1.4.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       December 2013

    Purpose
    =======
    DGETRF computes an LU factorization of a general M-by-N matrix A
    using partial pivoting with row interchanges.

    The factorization has the form
       A = P * L * U
    where P is a permutation matrix, L is lower triangular with unit
    diagonal elements (lower trapezoidal if m > n), and U is upper
    triangular (upper trapezoidal if m < n).

    This is the right-looking Level 3 BLAS version of the algorithm.

    Arguments
    =========
    NUM_GPUS
            (input) INTEGER
            The number of GPUS to be used for the factorization.

    M       (input) INTEGER
            The number of rows of the matrix A.  M >= 0.

    N       (input) INTEGER
            The number of columns of the matrix A.  N >= 0.

    A       (input/output) DOUBLE_PRECISION array on the GPU, dimension (LDDA,N).
            On entry, the M-by-N matrix to be factored.
            On exit, the factors L and U from the factorization
            A = P*L*U; the unit diagonal elements of L are not stored.

    LDDA     (input) INTEGER
            The leading dimension of the array A.  LDDA >= max(1,M).

    IPIV    (output) INTEGER array, dimension (min(M,N))
            The pivot indices; for 1 <= i <= min(M,N), row i of the
            matrix was interchanged with row IPIV(i).

    INFO    (output) INTEGER
            = 0:  successful exit
            < 0:  if INFO = -i, the i-th argument had an illegal value
                  or another error occured, such as memory allocation failed.
            > 0:  if INFO = i, U(i,i) is exactly zero. The factorization
                  has been completed, but the factor U is exactly
                  singular, and division by zero will occur if it is used
                  to solve a system of equations.
    =====================================================================    */

#define inAT(id,i,j) (d_lAT[(id)] + (i)*nb*lddat + (j)*nb)

    magma_int_t nb, n_local[MagmaMaxGPUs];
    magma_int_t maxm, mindim;
    magma_int_t i, j, d, lddat, lddwork;
    double *d_lAT[MagmaMaxGPUs];
    double *d_panel[MagmaMaxGPUs], *work;
    magma_queue_t streaml[MagmaMaxGPUs][2];

    /* Check arguments */
    *info = 0;
    if (m < 0)
        *info = -2;
    else if (n < 0)
        *info = -3;
    else if (ldda < max(1,m))
        *info = -5;

    if (*info != 0) {
        magma_xerbla( __func__, -(*info) );
        return *info;
    }

    /* Quick return if possible */
    if (m == 0 || n == 0)
        return *info;

    /* Function Body */
    mindim = min(m, n);
    nb     = magma_get_dgetrf_nb(m);

    if (nb <= 1 || nb >= n) {
        /* Use CPU code. */
        magma_dmalloc_cpu( &work, m * n );
        if ( work == NULL ) {
            *info = MAGMA_ERR_HOST_ALLOC;
            return *info;
        }
        magma_dgetmatrix( m, n, d_lA[0], ldda, work, m );
        lapackf77_dgetrf(&m, &n, work, &m, ipiv, info);
        magma_dsetmatrix( m, n, work, m, d_lA[0], ldda );
        magma_free_cpu(work);
    } else {
        /* Use hybrid blocked code. */
        maxm = ((m + 31)/32)*32;
        if( num_gpus > ceil((double)n/nb) ) {
            printf( " * too many GPUs for the matrix size, using %d GPUs\n", (int) num_gpus );
            *info = -1;
            return *info;
        }

        /* allocate workspace for each GPU */
        lddat = ((((((n+nb-1)/nb)/num_gpus)*nb)+31)/32)*32;
        lddat = (n+nb-1)/nb;                 /* number of block columns         */
        lddat = (lddat+num_gpus-1)/num_gpus; /* number of block columns per GPU */
        lddat = nb*lddat;                    /* number of columns per GPU       */
        lddat = ((lddat+31)/32)*32;          /* make it a multiple of 32        */
        for(i=0; i<num_gpus; i++){
            magma_setdevice(i);
            
            /* local-n and local-ld */
            n_local[i] = ((n/nb)/num_gpus)*nb;
            if (i < (n/nb)%num_gpus)
               n_local[i] += nb;
            else if (i == (n/nb)%num_gpus)
               n_local[i] += n%nb;
            
            /* workspaces */
            if (MAGMA_SUCCESS != magma_dmalloc( &d_panel[i], (3+num_gpus)*nb*maxm )) {
                for( j=0; j<=i; j++ ) {
                    magma_setdevice(j);
                }
                for( j=0; j<i; j++ ) {
                    magma_setdevice(j);
                    magma_free( d_panel[j] );
                    magma_free( d_lAT[j]   );
                }
                *info = MAGMA_ERR_DEVICE_ALLOC;
                return *info;
            }
            
            /* local-matrix storage */
            if (MAGMA_SUCCESS != magma_dmalloc( &d_lAT[i], lddat*maxm )) {
                for( j=0; j<=i; j++ ) {
                    magma_setdevice(j);
                    magma_free( d_panel[j] );
                }
                for( j=0; j<i; j++ ) {
                    magma_setdevice(j);
                    magma_free( d_lAT[j] );
                }
                *info = MAGMA_ERR_DEVICE_ALLOC;
                return *info;
            }
            
            /* create the streams */
            magma_queue_create( &streaml[i][0] );
            magma_queue_create( &streaml[i][1] );
            
            magmablasSetKernelStream(streaml[i][1]);
            magmablas_dtranspose2( d_lAT[i], lddat, d_lA[i], ldda, m, n_local[i] );
        }
        for(i=0; i<num_gpus; i++){
            magma_setdevice(i);
            cudaStreamSynchronize(streaml[i][0]);
            magmablasSetKernelStream(NULL);
        }
        magma_setdevice(0);

        /* cpu workspace */
        lddwork = maxm;
        if (MAGMA_SUCCESS != magma_dmalloc_pinned( &work, lddwork*nb*num_gpus )) {
            for(i=0; i<num_gpus; i++ ) {
                magma_setdevice(i);
                magma_free( d_panel[i] );
                magma_free( d_lAT[i]   );
            }
            *info = MAGMA_ERR_HOST_ALLOC;
            return *info;
        }

        /* calling multi-gpu interface with allocated workspaces and streams */
        //magma_dgetrf1_mgpu( num_gpus, m, n, nb, 0, d_lAT, lddat, ipiv, d_panel, work, maxm,
        //                   (magma_queue_t **)streaml, info );
        magma_dgetrf2_mgpu(num_gpus, m, n, nb, 0, d_lAT, lddat, ipiv, d_panel, work, maxm,
                           streaml, info);

        /* clean up */
        for( d=0; d<num_gpus; d++ ) {
            magma_setdevice(d);
            
            /* save on output */
            magmablas_dtranspose2( d_lA[d], ldda, d_lAT[d], lddat, n_local[d], m );
            magma_device_sync();
            magma_free( d_lAT[d]   );
            magma_free( d_panel[d] );
            magma_queue_destroy( streaml[d][0] );
            magma_queue_destroy( streaml[d][1] );
            magmablasSetKernelStream(NULL);
        } /* end of for d=1,..,num_gpus */
        magma_setdevice(0);
        magma_free_pinned( work );
    }
        
    return *info;
}

#undef inAT
