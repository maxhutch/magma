/*
    -- MAGMA (version 1.6.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date January 2015

       @author Mark Gates
       @generated from zlanhe.cu normal z -> d, Fri Jan 30 19:00:09 2015

*/
#include "common_magma.h"

#define inf_bs 32
#define max_bs 64

#define PRECISION_d


/* ====================================================================== */
/* inf-norm */

/* Computes row sums dwork[i] = sum( abs( A(i,:) )), i=0:n-1, for || A ||_inf,
 * where n is any size and A is stored lower.
 * Has ceil( n / inf_bs ) blocks of (inf_bs x 4) threads each (inf_bs=32).
 * z precision uses > 16 KB shared memory, so requires Fermi (arch >= 200). */
__global__ void
dlansy_inf_kernel_generic_lower(
    int n, const double* A, int lda, double *dwork,
    int n_full_block, int n_mod_bs )
{
#if (defined(PRECISION_s) || defined(PRECISION_d) || defined(PRECISION_c) || __CUDA_ARCH__ >= 200)
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    int diag = blockIdx.x*inf_bs;
    int ind  = blockIdx.x*inf_bs + tx;
    
    double res = 0.;
    
    __shared__ double la[inf_bs][inf_bs+1];
    
    if ( blockIdx.x < n_full_block ) {
        // ------------------------------
        // All full block rows
        A += ind;
        A += ty * lda;
        
        // ----------
        // loop over all blocks left of the diagonal block
        for(int i=0; i < diag; i += inf_bs ) {
            // 32x4 threads cooperatively load 32x32 block
            #pragma unroll 8
            for(int j=0; j < inf_bs; j += 4) {
                la[tx][ty+j] = A[j*lda];
            }
            A += lda*inf_bs;
            __syncthreads();
            
            // compute 4 partial sums of each row, i.e.,
            // for ty=0:  res = sum( la[tx, 0: 7] )
            // for ty=1:  res = sum( la[tx, 8:15] )
            // for ty=2:  res = sum( la[tx,16:23] )
            // for ty=3:  res = sum( la[tx,24:31] )
            #pragma unroll 8             
            for(int j=ty*8; j < ty*8 + 8; j++) {
                res += fabs( la[tx][j] );
            }
            __syncthreads();
        }
        
        // ----------
        // load diagonal block
        #pragma unroll 8
        for(int j=0; j < inf_bs; j += 4) {
            la[tx][ty+j] = A[j*lda];
        }
        __syncthreads();
        
        // copy lower triangle to upper triangle, and
        // make diagonal real (zero imaginary part)
        #pragma unroll 8
        for(int i=ty*8; i < ty*8 + 8; i++) {
            if ( i < tx ) {
                la[i][tx] = la[tx][i];
            }
            #if defined(PRECISION_z) || defined(PRECISION_c)
            else if ( i == tx ) {
                la[i][i] = MAGMA_D_MAKE( MAGMA_D_REAL( la[i][i] ), 0 );
            }
            #endif
        }
        __syncthreads();
        
        // partial row sums
        #pragma unroll 8
        for(int j=ty*8; j < ty*8 + 8; j++) {
            res += fabs( la[tx][j] );
        }
        __syncthreads();
        
        // ----------
        // loop over all 32x32 blocks below diagonal block
        A += inf_bs;
        for(int i=diag + inf_bs; i < n - n_mod_bs; i += inf_bs ) {
            // load block (transposed)
            #pragma unroll 8
            for(int j=0; j < inf_bs; j += 4) {
                la[ty+j][tx] = A[j*lda];
            }
            A += inf_bs;
            __syncthreads();
            
            // partial row sums
            #pragma unroll 8
            for(int j=ty*8; j < ty*8 + 8; j++) {
                res += fabs( la[tx][j] );
            }
            __syncthreads();
        }
        
        // ----------
        // last partial block, which is (n_mod_bs by inf_bs)
        if ( n_mod_bs > 0 ) {
            // load block (transposed), with zeros for rows outside matrix
            #pragma unroll 8
            for(int j=0; j < inf_bs; j += 4) {
                if ( tx < n_mod_bs ) {
                    la[ty+j][tx] = A[j*lda];
                }
                else {
                    la[ty+j][tx] = MAGMA_D_ZERO;
                }
            }
            __syncthreads();
            
            // partial row sums
            #pragma unroll 8
            for(int j=ty*8; j < ty*8 + 8; j++) {
                res += fabs( la[tx][j] );
            }
            __syncthreads();
        }
        
        // ----------
        // 32x4 threads store partial sums into shared memory
        la[tx][ty] = MAGMA_D_MAKE( res, 0. );
        __syncthreads();
        
        // first column of 32x1 threads computes final sum of each row
        if ( ty == 0 ) {
            res = res
                + MAGMA_D_REAL( la[tx][1] )
                + MAGMA_D_REAL( la[tx][2] )
                + MAGMA_D_REAL( la[tx][3] );
            dwork[ind] = res;
        }
    }
    else {
        // ------------------------------
        // Last, partial block row
        // Threads past end of matrix (i.e., ind >= n) are redundantly assigned
        // the last row (n-1). At the end, those results are ignored -- only
        // results for ind < n are saved into dwork.
        if ( tx < n_mod_bs ) {
            A += ind;
        }
        else {
            A += (blockIdx.x*inf_bs + n_mod_bs - 1);  // redundantly do last row
        }
        A += ty * lda;
        
        // ----------
        // loop over all blocks left of the diagonal block
        // each is (n_mod_bs by inf_bs)
        for(int i=0; i < diag; i += inf_bs ) {
            // load block
            #pragma unroll 8
            for(int j=0; j < inf_bs; j += 4) {
                la[tx][ty+j] = A[j*lda];
            }
            A += lda*inf_bs;
            __syncthreads();
            
            // partial row sums
            #pragma unroll 8
            for(int j=0; j < 8; j++) {
                res += fabs( la[tx][j+ty*8] );
            }
            __syncthreads();
        }
        
        // ----------
        // partial diagonal block
        if ( ty == 0 && tx < n_mod_bs ) {
            // sum rows left of diagonal
            for(int j=0; j < tx; j++) {
                res += fabs( *A );
                A += lda;
            }
            // sum diagonal (ignoring imaginary part)
            res += MAGMA_D_ABS( MAGMA_D_REAL( *A ));
            A += 1;
            // sum column below diagonal
            for(int j=tx+1; j < n_mod_bs; j++) {
                res += fabs( *A );
                A += 1;
            }
        }
        __syncthreads();
        
        // ----------
        // 32x4 threads store partial sums into shared memory
        la[tx][ty]= MAGMA_D_MAKE( res, 0. );
        __syncthreads();
        
        // first column of 32x1 threads computes final sum of each row
        // rows outside matrix are ignored
        if ( ty == 0 && tx < n_mod_bs ) {
            res = res
                + MAGMA_D_REAL( la[tx][1] )
                + MAGMA_D_REAL( la[tx][2] )
                + MAGMA_D_REAL( la[tx][3] );
            dwork[ind] = res;
        }
    }
#endif /* (PRECISION_s || PRECISION_d || PRECISION_c || __CUDA_ARCH__ >= 200) */
}



/* Computes row sums dwork[i] = sum( abs( A(i,:) )), i=0:n-1, for || A ||_inf,
 * where n is any size and A is stored upper.
 * Has ceil( n / inf_bs ) blocks of (inf_bs x 4) threads each (inf_bs=32).
 * z precision uses > 16 KB shared memory, so requires Fermi (arch >= 200).
 * The upper implementation is similar to lower, but processes blocks
 * in the transposed order:
 * lower goes from left over to diagonal, then down to bottom;
 * upper goes from top  down to diagonal, then over to right.
 * Differences are noted with # in comments. */
__global__ void
dlansy_inf_kernel_generic_upper(
    int n, const double* A, int lda, double *dwork,
    int n_full_block, int n_mod_bs )
{
#if (defined(PRECISION_s) || defined(PRECISION_d) || defined(PRECISION_c) || __CUDA_ARCH__ >= 200)
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    int diag = blockIdx.x*inf_bs;
    int ind  = blockIdx.x*inf_bs + tx;
    
    double res = 0.;
    
    __shared__ double la[inf_bs][inf_bs+1];
    
    if ( blockIdx.x < n_full_block ) {
        // ------------------------------
        // All full block #columns
        A += blockIdx.x*inf_bs*lda + tx;               //#
        A += ty * lda;
        
        // ----------
        // loop over all blocks #above the diagonal block
        for(int i=0; i < diag; i += inf_bs ) {
            // 32x4 threads cooperatively load 32x32 block (#transposed)
            #pragma unroll 8
            for(int j=0; j < inf_bs; j += 4) {
                la[ty+j][tx] = A[j*lda];               //#
            }
            A += inf_bs;                               //#
            __syncthreads();
            
            // compute 4 partial sums of each row, i.e.,
            // for ty=0:  res = sum( la[tx, 0: 7] )
            // for ty=1:  res = sum( la[tx, 8:15] )
            // for ty=2:  res = sum( la[tx,16:23] )
            // for ty=3:  res = sum( la[tx,24:31] )
            #pragma unroll 8             
            for(int j=ty*8; j < ty*8 + 8; j++) {
                res += fabs( la[tx][j] );
            }
            __syncthreads();
        }
        
        // ----------
        // load diagonal block
        #pragma unroll 8
        for(int j=0; j < inf_bs; j += 4) {
            la[tx][ty+j] = A[j*lda];
        }
        __syncthreads();
        
        // copy #upper triangle to #lower triangle, and
        // make diagonal real (zero imaginary part)
        #pragma unroll 8
        for(int i=ty*8; i < ty*8 + 8; i++) {
            if ( i > tx ) {                            //#
                la[i][tx] = la[tx][i];
            }
            #if defined(PRECISION_z) || defined(PRECISION_c)
            else if ( i == tx ) {
                la[i][i] = MAGMA_D_MAKE( MAGMA_D_REAL( la[i][i] ), 0 );
            }
            #endif
        }
        __syncthreads();
        
        // partial row sums
        #pragma unroll 8
        for(int j=ty*8; j < ty*8 + 8; j++) {
            res += fabs( la[tx][j] );
        }
        __syncthreads();
        
        // ----------
        // loop over all 32x32 blocks #right of diagonal block
        A += inf_bs*lda;                               //#
        for(int i=diag + inf_bs; i < n - n_mod_bs; i += inf_bs ) {
            // load block (#non-transposed)
            #pragma unroll 8
            for(int j=0; j < inf_bs; j += 4) {
                la[tx][ty+j] = A[j*lda];               //#
            }
            A += inf_bs*lda;                           //#
            __syncthreads();
            
            // partial row sums
            #pragma unroll 8
            for(int j=ty*8; j < ty*8 + 8; j++) {
                res += fabs( la[tx][j] );
            }
            __syncthreads();
        }
        
        // ----------
        // last partial block, which is #(inf_bs by n_mod_bs)
        if ( n_mod_bs > 0 ) {
            // load block (#non-transposed), with zeros for #cols outside matrix
            #pragma unroll 8
            for(int j=0; j < inf_bs; j += 4) {
                if ( ty+j < n_mod_bs ) {               //#
                    la[tx][ty+j] = A[j*lda];           //#
                }
                else {
                    la[tx][ty+j] = MAGMA_D_ZERO;       //#
                }
            }
            __syncthreads();
            
            // partial row sums
            #pragma unroll 8
            for(int j=ty*8; j < ty*8 + 8; j++) {
                res += fabs( la[tx][j] );
            }
            __syncthreads();
        }
        
        // ----------
        // 32x4 threads store partial sums into shared memory
        la[tx][ty] = MAGMA_D_MAKE( res, 0. );
        __syncthreads();
        
        // first column of 32x1 threads computes final sum of each row
        if ( ty == 0 ) {
            res = res
                + MAGMA_D_REAL( la[tx][1] )
                + MAGMA_D_REAL( la[tx][2] )
                + MAGMA_D_REAL( la[tx][3] );
            dwork[ind] = res;
        }
    }
    else {
        // ------------------------------
        // Last, partial block #column
        // Instead of assigning threads ind >= n to the last row (n-1), as in Lower,
        // Upper simply adjusts loop bounds to avoid loading columns outside the matrix.
        // Again, at the end, those results are ignored -- only
        // results for ind < n are saved into dwork.
        A += blockIdx.x*inf_bs*lda + tx;               //#
        A += ty * lda;
        
        // ----------
        // loop over all blocks #above the diagonal block
        // each is #(inf_bs by n_mod_bs)
        for(int i=0; i < diag; i += inf_bs ) {
            // load block (#transposed), #ignoring columns outside matrix
            #pragma unroll 8
            for(int j=0; j < inf_bs; j += 4) {
                if ( ty+j < n_mod_bs ) {
                    la[ty+j][tx] = A[j*lda];
                }
            }
            A += inf_bs;                               //#
            __syncthreads();
            
            // partial row sums
            #pragma unroll 8
            for(int j=0; j < 8; j++) {
                res += fabs( la[tx][j+ty*8] );
            }
            __syncthreads();
        }
        
        // ----------
        // partial diagonal block
        if ( ty == 0 && tx < n_mod_bs ) {
            // #transpose pointer within diagonal block
            // #i.e., from A = A(tx,ty), transpose to A = A(ty,tx).
            A = A - tx - ty*lda + tx*lda + ty;
            
            // sum #column above diagonal
            for(int j=0; j < tx; j++) {
                res += fabs( *A );
                A += 1;                                //#
            }
            // sum diagonal (ignoring imaginary part)
            res += MAGMA_D_ABS( MAGMA_D_REAL( *A ));
            A += lda;                                  //#
            // sum #row right of diagonal
            for(int j=tx+1; j < n_mod_bs; j++) {
                res += fabs( *A );
                A += lda;                              //#
            }
        }
        __syncthreads();
        
        // ----------
        // 32x4 threads store partial sums into shared memory
        la[tx][ty]= MAGMA_D_MAKE( res, 0. );
        __syncthreads();
        
        // first column of 32x1 threads computes final sum of each row
        // rows outside matrix are ignored
        if ( ty == 0 && tx < n_mod_bs ) {
            res = res
                + MAGMA_D_REAL( la[tx][1] )
                + MAGMA_D_REAL( la[tx][2] )
                + MAGMA_D_REAL( la[tx][3] );
            dwork[ind] = res;
        }
    }
#endif /* (PRECISION_s || PRECISION_d || PRECISION_c || __CUDA_ARCH__ >= 200) */
}


/* Computes row sums dwork[i] = sum( abs( A(i,:) )), i=0:n-1, for || A ||_inf */
extern "C" void
dlansy_inf(
    magma_uplo_t uplo, int n,
    magmaDouble_const_ptr A, int lda,
    magmaDouble_ptr dwork )
{
    int blocks = (n - 1)/inf_bs + 1;
    dim3 grid(blocks, 1, 1);
    dim3 threads(inf_bs, 4, 1);

    int n_full_block = (n - n % inf_bs) /inf_bs;
    int n_mod_bs = n % inf_bs;
    if ( uplo == MagmaLower) {
        dlansy_inf_kernel_generic_lower<<< grid, threads, 0, magma_stream >>>
            ( n, A, lda, dwork, n_full_block, n_mod_bs );
    }
    else {
        dlansy_inf_kernel_generic_upper<<< grid, threads, 0, magma_stream >>>
            ( n, A, lda, dwork, n_full_block, n_mod_bs );
    }
}


/* ====================================================================== */
/* max-norm */

/* Computes dwork[i] = max( abs( A(i,0:i) )), i=0:n-1, for ||A||_max, where A is stored lower */
__global__ void
dlansy_max_kernel_lower(
    int n, const double* A, int lda, double *dwork )
{
    int ind = blockIdx.x*max_bs + threadIdx.x;
    double res = 0;

    if (ind < n) {
        A += ind;
        for(int j=0; j < ind; ++j) {
            res = fmax( res, fabs( *A ));
            A += lda;
        }
        // diagonal element (ignoring imaginary part)
        res = fmax( res, MAGMA_D_ABS( MAGMA_D_REAL( *A )));
        dwork[ind] = res;
    }
}


/* Computes dwork[i] = max( abs( A(i,0:i) )), i=0:n-1, for ||A||_max, where A is stored upper. */
__global__ void
dlansy_max_kernel_upper(
    int n, const double* A, int lda, double *dwork )
{
    int ind = blockIdx.x*max_bs + threadIdx.x;
    double res = 0;

    if (ind < n) {
        A += ind;
        A += (n-1)*lda;
        for(int j=n-1; j > ind; j--) {
            res = fmax( res, fabs( *A ));
            A -= lda;
        }
        // diagonal element (ignoring imaginary part)
        res = fmax( res, MAGMA_D_ABS( MAGMA_D_REAL( *A )));
        dwork[ind] = res;
    }
}


/* Computes dwork[i] = max( abs( A(i,:) )), i=0:n-1, for ||A||_max */
extern "C" void
dlansy_max(
    magma_uplo_t uplo, int n,
    magmaDouble_const_ptr A, int lda,
    magmaDouble_ptr dwork )
{
    int blocks = (n - 1)/max_bs + 1;
    dim3 grid(blocks, 1, 1);
    dim3 threads(max_bs, 1, 1);

    if ( uplo == MagmaLower ) {
        dlansy_max_kernel_lower<<< grid, threads, 0, magma_stream >>>
            ( n, A, lda, dwork );
    }
    else {
        dlansy_max_kernel_upper<<< grid, threads, 0, magma_stream >>>
            ( n, A, lda, dwork );
    }
}


/* ====================================================================== */
/**
    Purpose
    -------
    DLANSY returns the value of the one norm, or the Frobenius norm, or
    the infinity norm, or the element of largest absolute value of a
    real symmetric matrix A.
    
       DLANSY = ( max(abs(A(i,j))), NORM = 'M' or 'm'
                (
                ( norm1(A),         NORM = '1', 'O' or 'o'      ** supported only for (PRECISION_s || PRECISION_d || PRECISION_c || __CUDA_ARCH__ >= 200)
                (
                ( normI(A),         NORM = 'I' or 'i'           ** supported only for (PRECISION_s || PRECISION_d || PRECISION_c || __CUDA_ARCH__ >= 200)
                (
                ( normF(A),         NORM = 'F', 'f', 'E' or 'e' ** not yet supported
    
    where norm1 denotes the one norm of a matrix (maximum column sum),
    normI denotes the infinity norm of a matrix (maximum row sum) and
    normF denotes the Frobenius norm of a matrix (square root of sum of squares).
    Note that max(abs(A(i,j))) is not a consistent matrix norm.
    
    Returns DLANSY < 0: if DLANSY = -i, the i-th argument had an illegal value.
    
    Arguments:
    ----------
    @param[in]
    norm    CHARACTER*1
            Specifies the value to be returned in DLANSY as described above.
    
    @param[in]
    uplo    magma_uplo_t
            Specifies whether the upper or lower triangular part of the
            symmetric matrix A is to be referenced.
      -     = MagmaUpper: Upper triangular part of A is referenced
      -     = MagmaLower: Lower triangular part of A is referenced
    
    @param[in]
    n       INTEGER
            The order of the matrix A. N >= 0. When N = 0, DLANSY is
            set to zero.
    
    @param[in]
    A       DOUBLE PRECISION array on the GPU, dimension (LDA,N)
            The symmetric matrix A. If UPLO = MagmaUpper, the leading n by n
            upper triangular part of A contains the upper triangular part
            of the matrix A, and the strictly lower triangular part of A
            is not referenced. If UPLO = MagmaLower, the leading n by n lower
            triangular part of A contains the lower triangular part of
            the matrix A, and the strictly upper triangular part of A is
            not referenced. Note that the imaginary parts of the diagonal
            elements need not be set and are assumed to be zero.
    
    @param[in]
    lda     INTEGER
            The leading dimension of the array A. LDA >= max(N,1).
    
    @param
    dwork   (workspace) DOUBLE PRECISION array on the GPU, dimension (MAX(1,LWORK)),
            where LWORK >= N.
            NOTE: this is different than LAPACK, where WORK is required
            only for norm1 and normI. Here max-norm also requires work.
    
    @ingroup magma_daux2
    ********************************************************************/

extern "C" double
magmablas_dlansy(
    magma_norm_t norm, magma_uplo_t uplo, magma_int_t n,
    magmaDouble_const_ptr dA, magma_int_t ldda,
    magmaDouble_ptr dwork )
{
    magma_int_t info = 0;
    magma_int_t arch = magma_getdevice_arch();
    // 1-norm == inf-norm since A is symmetric
    bool inf_norm = (norm == MagmaInfNorm || norm == MagmaOneNorm);
    bool max_norm = (norm == MagmaMaxNorm);
    
    // inf_norm Double-Complex requires > 16 KB shared data (arch >= 200)
    #if defined(PRECISION_z)
    const bool inf_implemented = (magma_getdevice_arch() >= 200);
    #else
    const bool inf_implemented = true;
    #endif
    
    if ( ! (max_norm || (inf_norm && inf_implemented)) )
        info = -1;
    else if ( uplo != MagmaUpper && uplo != MagmaLower )
        info = -2;
    else if ( n < 0 )
        info = -3;
    else if ( ldda < n )
        info = -5;
    
    if ( info != 0 ) {
        magma_xerbla( __func__, -(info) );
        return info;
    }
    
    /* Quick return */
    if ( n == 0 )
        return 0;
        
    double res = 0;
    if ( inf_norm ) {
        dlansy_inf( uplo, n, dA, ldda, dwork );
    }
    else {
        dlansy_max( uplo, n, dA, ldda, dwork );
    }
    int i = magma_idamax( n, dwork, 1 ) - 1;
    cudaMemcpy( &res, &dwork[i], sizeof(double), cudaMemcpyDeviceToHost );
    
    return res;
}
