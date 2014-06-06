/*
    -- MAGMA (version 1.4.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       December 2013

       @precisions normal z -> s d c

*/
#include "common_magma.h"

#define inf_bs 32
#define max_bs 64


/* ====================================================================== */
/* inf-norm */

/* Computes row sums dwork[i] = sum( abs( A(i,:) )), i=0:n-1, for || A ||_inf,
 * where n % inf_bs == 0 and A is stored lower.
 * Has ceil( n / inf_bs ) blocks of (inf_bs x 4) threads each.
 */
__global__ void
zlanhe_inf_kernel_special_l(
    int n, const magmaDoubleComplex* A, int lda, double *dwork )
{
#if (__CUDA_ARCH__ >= 200)
    int tx  = threadIdx.x;
    int ty  = threadIdx.y;
    int ind = blockIdx.x*inf_bs + tx;
    double res = 0.;
    
    __shared__ magmaDoubleComplex la[inf_bs][inf_bs+1];
    
    A += ind;
    A += ty * lda;
    int break_d = blockIdx.x*inf_bs;
    
    // loop over all 32x32 blocks left of the diagonal block
    for(int i=0; i < break_d; i += inf_bs ) {
        // 32x4 threads cooperatively load 32x32 block
        #pragma unroll 8
        for(int j=0; j < inf_bs; j += 4) {
            la[tx][ty+j] = A[j*lda];
        }
        __syncthreads();
        
        // compute 4 partial sums of each row
        #pragma unroll 8
        for(int j=0; j < 8; j++) {
            res += cuCabs( la[tx][j+ty*8] );
        }
        A += lda*inf_bs;
        __syncthreads();
    }
    
    // 32x4 threads cooperatively load 32x32 diagonal block
    #pragma unroll 8
    for(int j=0; j < inf_bs; j += 4)
        la[ty+j][tx] = A[j*lda];
    A += inf_bs;
    __syncthreads();
    
    // symmetrize block
    // TODO make diagonal element real
    #pragma unroll 8
    for(int i=ty*8; i < (1+ty)*inf_bs/4; i++) {
        if ( i < tx ) {
            la[tx][i] = la[i][tx];
        }
        else
            la[tx][i] = la[tx][i];  // TODO: not needed
    }
    __syncthreads();
    
    // compute 4 partial sums of each row
    #pragma unroll 8
    for(int j=0; j < inf_bs/4; j++) {
        res += cuCabs( la[tx][j+ty*8] );
    }
    break_d += inf_bs;
    __syncthreads();
    
    // loop over all 32x32 blocks below diagonal block
    for(int i=break_d; i < n; i += inf_bs ) {
        // 32x4 threads cooperatively load 32x32 block
        #pragma unroll 8
        for(int j=0; j < inf_bs; j += 4)
            la[ty+j][tx] = A[j*lda];
        A += inf_bs;
        __syncthreads();
        
        // compute 4 partial sums of each row
        #pragma unroll 8
        for(int j=0; j < inf_bs/4; j++) {
            res += cuCabs( la[tx][j+ty*8] );
        }
        __syncthreads();
    }
    
    // store partial sums into shared memory
    la[tx][ty] = MAGMA_Z_MAKE( res, 0. );
    __syncthreads();
    
    // 32x1 threads compute final result of each row
    if ( ty == 0 ) {
        res = res
            + MAGMA_Z_REAL( la[tx][1] )
            + MAGMA_Z_REAL( la[tx][2] )
            + MAGMA_Z_REAL( la[tx][3] );
        dwork[ind] = res;
    }
#endif /* (__CUDA_ARCH__ >= 200) */
}


/* Computes row sums dwork[i] = sum( abs( A(i,:) )), i=0:n-1, for || A ||_inf,
 * where n is any size and A is stored lower */
__global__ void
zlanhe_inf_kernel_generic_l(
    int n, const magmaDoubleComplex* A, int lda, double *dwork,
    int n_full_block, int n_mod_bs )
{
#if (__CUDA_ARCH__ >= 200)
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    int ind = blockIdx.x*inf_bs + tx;
    
    double res = 0.;
    
    __shared__ magmaDoubleComplex la[inf_bs][inf_bs+1];
    
    if ( blockIdx.x == n_full_block ) {
        /************************************************************************
        -- Last (partial) block --
        -- We will do something unusual here
        -- Threads past end of matrix (i.e., ind >= n) are redundantly assigned 
        -- the last row (n-1). At the end, those results are ignored -- only
        -- results for ind < n are saved into dwork.
        -- For sufficiently large matrix the overhead will be very low
        *************************************************************************/
        if ( tx < n_mod_bs ) {
            A += ( blockIdx.x*inf_bs + tx );
        }
        else {
            A += ( blockIdx.x*inf_bs + n_mod_bs - 1);  // redundantly do last row
        }
        A += ty * lda;
        int break_d = blockIdx.x*inf_bs;
    
        /*----------------------------
            Go Right
        -------------------------------*/
        for(int i=0; i < break_d; i += inf_bs ) {
            #pragma unroll 8
            for(int j=0; j < inf_bs; j += 4) {
                la[tx][ty+j] = A[j*lda];
            }
            __syncthreads();
            
            #pragma unroll 8
            for(int j=0; j < 8; j++) {
                res += cuCabs( la[tx][j+ty*8] );
            }
            A += lda*inf_bs;
            __syncthreads();
        }
        
        /* we don't need to make results for rows >= n zero, as those computation will be discarded. */
        
        if ( ty == 0 ) {
            /*--------------------------------------------
                he will compute the triangular parts
                others will be waiting with values.
            -----------------------------------------------*/
            int j;
            int count = 1;  // TODO don't need initialization
            if ( tx < n_mod_bs )
                count = tx;
            else
                count = n_mod_bs;
            for(j=0; j <= count; j++) {
                res += cuCabs( A[j*lda] );
            }
            A += tx*lda;
            count = 1;
            for( ; j < n_mod_bs; j++) {
                res += cuCabs( A[count] );
                count++;
            }
        }
        __syncthreads();
        
        la[tx][ty]= MAGMA_Z_MAKE( res, 0. );
        __syncthreads();
        
        /*--------------------------------------------------------
        The leader accumulates all the results from his peer.
        ----------------------------------------------------------*/
        if ( ty == 0 ) {
            res = res
                + MAGMA_Z_REAL( la[tx][1] )
                + MAGMA_Z_REAL( la[tx][2] )
                + MAGMA_Z_REAL( la[tx][3] );
            if ( tx < n_mod_bs )
                dwork[ind] = res;
        }
    }
    else {
        /*-----------------------------------
        -- All the blocks but the last one --
        -------------------------------------*/
        A += ind;
        A += ty * lda;
        int break_d = blockIdx.x*inf_bs;
        
        /*----------------------------
            Go Right
        -------------------------------*/
        for(int i=0; i < break_d; i += inf_bs ) {
            #pragma unroll 8
            for(int j=0; j < inf_bs; j += 4) {
                la[tx][ty+j] = A[j*lda];
            }
            __syncthreads();
            
            #pragma unroll 8
            for(int j=0; j < 8; j++) {
                res += cuCabs( la[tx][j+ty*8] );
            }
            A += lda*inf_bs;
            __syncthreads();
        }

        /*------------------------------------
            Diagonal
            Copy + Transpose lower triangle
        --------------------------------------*/
        #pragma unroll 8
        for(int j=0; j < inf_bs; j += 4)
            la[ty+j][tx] = A[j*lda];
        
        A += inf_bs;
        __syncthreads();
        
        /*--------------------------------------------
            Mirror Upper Triangle to Lower triangle
        ---------------------------------------------*/
        #pragma unroll 8
        for(int i=ty*8; i < (1+ty)*inf_bs/4; i++) {
            if ( i < tx ) {
                la[tx][i] = la[i][tx];
            }
            else
                la[tx][i] = la[tx][i];  // TODO: not needed
        }
        __syncthreads();
        
        /*--------------------------------
            Do diagonal Computation
        -----------------------------------*/
        #pragma unroll 8
        for(int j=0; j < inf_bs/4; j++) {
            res += cuCabs( la[tx][j+ty*8] );
        }
        break_d += inf_bs;
        __syncthreads();
        
        n -= n_mod_bs;
        
        /*-----------------------------
            Go Down
        -------------------------------*/
        for(int i=break_d; i < n; i += inf_bs ) {
            #pragma unroll 8
            for(int j=0; j < inf_bs; j += 4)
                la[ty+j][tx] = A[j*lda];
            A += inf_bs;
            __syncthreads();
            
            #pragma unroll 8
            for(int j=0; j < inf_bs/4; j++) {
                res += cuCabs( la[tx][j+ty*8] );
            }
            __syncthreads();
        }
        
        /*---------------------------------------------
            doing n_mod_bs stuffs here.
            Symmetric is giving us benefit .. true
        -----------------------------------------------*/
        A -= tx;
        if ( tx < n_mod_bs ) {
            A += tx;
        }
        else {
            A += (n_mod_bs-1); /* Same as above */
        }
        
        #pragma unroll 8
        for(int j=0; j < inf_bs; j += 4) {
            if ( tx < n_mod_bs )
                la[ty+j][tx] = A[j*lda];      //MAGMA_Z_MUL( MAGMA_Z_ONE,  A[j*lda] );  // huh? just A[j*lda]?
            else
                la[ty+j][tx] = MAGMA_Z_ZERO;  //MAGMA_Z_MUL( MAGMA_Z_ZERO, A[j*lda] );  // huh? just 0?
        }
        __syncthreads();
        
        /*----------------------------------------
            What about doing some Zeroing here?
            instead of zeroing before?
        -----------------------------------------*/
        #pragma unroll 8
        for(int j=0; j < inf_bs/4; j++) {
            res += cuCabs( la[tx][j+ty*8] );
        }
        __syncthreads();
        
        la[tx][ty] = MAGMA_Z_MAKE( res, 0. );
        __syncthreads();
        
        /*--------------------------------------------------------
            The leader accumulates all the results from his peer.
        ----------------------------------------------------------*/
        if ( ty == 0 ) {
            res = res
                + MAGMA_Z_REAL( la[tx][1] )
                + MAGMA_Z_REAL( la[tx][2] )
                + MAGMA_Z_REAL( la[tx][3] );
            dwork[ind] = res;
        }
    }
#endif /* (__CUDA_ARCH__ >= 200) */
}


/* Computes row sums dwork[i] = sum( abs( A(i,:) )), i=0:n-1, for || A ||_inf,
 * where n is any size and A is stored upper */
__global__ void
zlanhe_inf_kernel_generic_u(
    int n, const magmaDoubleComplex* A, int lda, double *dwork,
    int n_full_block, int n_mod_bs )
{
#if (__CUDA_ARCH__ >= 200)
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    int ind = blockIdx.x*inf_bs + tx;
    
    double res = 0.;
    
    __shared__ magmaDoubleComplex la[inf_bs][inf_bs+1];
    int blockIdxx = blockIdx.x;
    
    if ( blockIdx.x == n_full_block ) {
        /************************************************************************
        -- Last block --
        -- We will do something unusual here
        -- For sufficiently large matrix the overhead will be very low
        *************************************************************************/
        
        ind = tx;
        A += lda*(n-1);
        
        if ( tx < n_mod_bs ) {
            A += tx;
        }
        else {
            A += (n_mod_bs - 1);
        }
        A -= ty * lda;
        int break_d = blockIdx.x*inf_bs;
        
        /*----------------------------
            Go Right
        -------------------------------*/
        for(int i=0; i < break_d; i += inf_bs ) {
            #pragma unroll 8
            for(int j=0; j < inf_bs; j += 4) {
                la[tx][ty+j] = A[-j*lda];
            }
            __syncthreads();
            
            #pragma unroll 8
            for(int j=0; j < 8; j++) {
                res += cuCabs( la[tx][j+ty*8] );
            }
            A -= lda*inf_bs;
            __syncthreads();
        }
        
        /* we don't need to make zero, as those computation will be discarded. */
        if ( ty == 0 ) {
            /*--------------------------------------------
                he will compute the triangular parts
                others will be waiting with values.
            -----------------------------------------------*/
            int j;
            int count = 1;
            if ( tx < n_mod_bs )
                count = n_mod_bs- tx;
            else
                count = n_mod_bs;
            for(j=0; j < count; j++) {
                res += cuCabs( A[-j*lda] );
            }
            A -= (count-1)*lda;
            count = 1;
            for( ; j < n_mod_bs; j++) {
                res += cuCabs( A[-count] );
                    count++;
            }
        }
        else {
        }
        __syncthreads();
        
        la[tx][ty] = MAGMA_Z_MAKE( res, 0. );
        __syncthreads();
        
        /*--------------------------------------------------------
            The leader accumulates all the results from his peer.
        ----------------------------------------------------------*/
        if ( ty == 0 ) {
            res = res
                + MAGMA_Z_REAL( la[tx][1] )
                + MAGMA_Z_REAL( la[tx][2] )
                + MAGMA_Z_REAL( la[tx][3] );
            if ( tx < n_mod_bs )
                dwork[ind] = res;
        }
    }
    else {
        /*-----------------------------------
        -- All the blocks but the last one --
        -- By the way this code can be optimized more.
        -------------------------------------*/
        ind = blockIdx.x*inf_bs + tx + n_mod_bs;
        const magmaDoubleComplex *A1 = A;
        A += lda*(n-1);
        
        A += ind;
        A -= ty * lda;
        
        int break_d = (n/inf_bs - blockIdxx - 1)*inf_bs;
        /*----------------------------
            Go Left
        -------------------------------*/
        for(int i=0; i < break_d; i += inf_bs ) {
            #pragma unroll 8
            for(int j=0; j < inf_bs; j += 4) {
                la[tx][ty+j] = A[-j*lda];
            }
            __syncthreads();
            
            #pragma unroll 8
            for(int j=0; j < 8; j++) {
                res += cuCabs( la[tx][j+ty*8] );
            }
            A -= lda*inf_bs;
            __syncthreads();
        }
        
        /*------------------------------------
            Diagonal
            Copy + Transpose lower triangle
        --------------------------------------*/
        #pragma unroll 8
        for(int j=0; j < inf_bs; j += 4) {
            la[tx][31-ty-j] = A[ -j * lda];
        }
        
        A -= inf_bs;
        __syncthreads();
        
        /*--------------------------------------------
            Mirror Upper Triangle to Lower triangle
        ---------------------------------------------*/
        #pragma unroll 8
        for(int i=ty*8; i < (1+ty)*inf_bs/4; i++) {
            if ( i < tx ) {
                la[tx][i] = la[i][tx];
            }
            else {
                la[tx][i] = la[tx][i];  // TODO: not needed
            }
        }
        __syncthreads();
        
        /*--------------------------------
            Do diagonal Computation
        -----------------------------------*/
        #pragma unroll 8
        for(int j=0; j < inf_bs/4; j++) {
            res += cuCabs( la[tx][j+ty*8] );
        }
        break_d += inf_bs;
        __syncthreads();
        
        n -= n_mod_bs;
        /*-----------------------------
            Go Up
        -------------------------------*/
        int i;
        for( i=break_d; i < n; i += inf_bs ) {
            #pragma unroll 8
            for(int j=0; j < inf_bs; j += 4) {
                la[ty+j][tx] = A[- j * lda];
            }
            A -= inf_bs;
            __syncthreads();
            
            #pragma unroll 8
            for(int j=0; j < inf_bs/4; j++) {
                res += cuCabs ( la[31-tx][j+ty*8] );
            }
            __syncthreads();
        }
        
        /*---------------------------------------------
            doing n_mod_bs stuffs here.
            Symmetric is giving us benefit .. true
            Do the other way please......
            see zlanhe_inf_kernel_generic_l code above
            TODO compare performance with lower case and use that implementation if better.
        -----------------------------------------------*/
        A1 = A1 + n_mod_bs*lda + tx*lda;
        if ( ty == 0 ) {
            for( int j = 0; j < n_mod_bs; j++) {
                res += cuCabs( A1[ j + lda * blockIdx.x * inf_bs ] );
            }
        }
        __syncthreads();
        
        la[tx][ty]= MAGMA_Z_MAKE( res, 0);
        __syncthreads();
        
        /*--------------------------------------------------------
            The leader accumulates all the results from his peer.
        ----------------------------------------------------------*/
        if ( ty == 0 ) {
            res = res
                + MAGMA_Z_REAL( la[tx][1] )
                + MAGMA_Z_REAL( la[tx][2] )
                + MAGMA_Z_REAL( la[tx][3] );
            dwork[ind] = res;
        }
    }
#endif /* (__CUDA_ARCH__ >= 200) */
}


/* Computes row sums dwork[i] = sum( abs( A(i,:) )), i=0:n-1, for || A ||_inf,
 * where n % inf_bs == 0 and A is stored upper */
__global__ void
zlanhe_inf_kernel_special_u(
    int n, const magmaDoubleComplex* A, int lda, double *dwork )
{
#if (__CUDA_ARCH__ >= 200)
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int ind = blockIdx.x*inf_bs + tx;
    double res = 0.;
    
    /*
        Reverse Computation ...
        - Left
        - Triangle
        - Up
    */
    
    A += lda*(n-1);
    __shared__ magmaDoubleComplex la[inf_bs][inf_bs+1];
    
    A += ind;
    A -= ty * lda;
    int break_d = (n / inf_bs - blockIdx.x-1 )*inf_bs;
    
    for(int i=0; i < break_d; i += inf_bs ) {
        #pragma unroll 8
        for(int j=0; j < inf_bs; j += 4) {
            la[tx][ty+j] = A[-j*lda];
        }
        __syncthreads();
        
        #pragma unroll 8
        for(int j=0; j < 8; j++) {
            res += cuCabs( la[tx][j+ty*8] );
        }
        A -= lda*inf_bs;
        __syncthreads();
    }
    
    #pragma unroll 8
    for(int j=0; j < inf_bs; j += 4)
        la[tx][31-ty-j] = A[ -j * lda];
    
    /* Look at the indexing changes */
    A -= inf_bs;
    __syncthreads();
    
    #pragma unroll 8
    for(int i=ty*8; i < (1+ty)*inf_bs/4; i++) {
        if ( i < tx ) {
            la[tx][i] = la[i][tx];
        }
        else {
            la[tx][i] = la[tx][i];  // TODO: not needed
        }
    }
    __syncthreads();
    
    #pragma unroll 8
    for(int j=0; j < inf_bs/4; j++) {
        res += cuCabs( la[tx][j+ty*8] );
    }
    break_d += inf_bs;
    __syncthreads();
    
    for(int i=break_d; i < n; i += inf_bs ) {
        #pragma unroll 8
        for(int j=0; j < inf_bs; j += 4)
            la[ty+j][tx] = A[ -j * lda];
        A -= inf_bs;
        __syncthreads();
        
        #pragma unroll 8
        for(int j=0; j < inf_bs/4; j++) {
            res += cuCabs( la[31-tx][j+ty*8] );
        }
        __syncthreads();
    }
    
    la[tx][ty]= MAGMA_Z_MAKE( res, 0. );
    __syncthreads();
    
    if ( ty == 0 ) {
        res = res
            + MAGMA_Z_REAL( la[tx][1] )
            + MAGMA_Z_REAL( la[tx][2] )
            + MAGMA_Z_REAL( la[tx][3] );
        dwork[ind] = res;
    }
#endif /* (__CUDA_ARCH__ >= 200) */
}


/* Computes row sums dwork[i] = sum( abs( A(i,:) )), i=0:n-1, for || A ||_inf */
extern "C" void
zlanhe_inf(
    magma_uplo_t uplo, int n, const magmaDoubleComplex *A, int lda, double *dwork )
{
    /* Note: The UPLO = 'U' Version can be optimized more. */
    int blocks = (n - 1)/inf_bs + 1;
    dim3 grid(blocks, 1, 1);
    dim3 threads(inf_bs, 4, 1);

    if ( n % inf_bs == 0 ) {
        if ( uplo == 'L' || uplo == 'l') {
            zlanhe_inf_kernel_special_l<<< grid, threads, 0, magma_stream >>>
                ( n, A, lda, dwork );
        }
        else {
            zlanhe_inf_kernel_special_u<<< grid, threads, 0, magma_stream >>>
                ( n, A, lda, dwork);
        }
    }
    else {
        int n_full_block = (n - n % inf_bs) /inf_bs;
        int n_mod_bs = n % inf_bs;
        if ( uplo == 'L' || uplo == 'l') {
            zlanhe_inf_kernel_generic_l<<< grid, threads, 0, magma_stream >>>
                ( n, A, lda, dwork, n_full_block, n_mod_bs );
        }
        else {
            zlanhe_inf_kernel_generic_u<<< grid, threads, 0, magma_stream >>>
                ( n, A, lda, dwork, n_full_block, n_mod_bs );
        }
    }
}


/* ====================================================================== */
/* max-norm */

/* Computes dwork[i] = max( abs( A(i,0:i) )), i=0:n-1, for ||A||_max, where A is stored lower */
__global__ void
zlanhe_max_kernel_l(
    int n, const magmaDoubleComplex* A, int lda, double *dwork )
{
    int tx  = threadIdx.x;
    int ind = blockIdx.x * max_bs + tx;
    double res = 0., res1;

    int break_d = blockIdx.x * max_bs;

    if (ind < n) {
        A += ind;
        
        // loop over blocks left of diagonal block
        for(int i=0; i < break_d; i += max_bs ) {
            #pragma unroll 8
            for(int j=0; j < max_bs; j++) {
                res1 = cuCabs( A[j*lda] );
                res = fmax( res, res1 );
            }
            
            A += lda*max_bs;
        }
        
        // process diagonal block
        for(int j=0; j <= tx; j++) {
            res1 = cuCabs( A[j*lda] );
            res = fmax( res, res1 );
        }
        
        dwork[ind] = res;
    }
}


/* Computes dwork[i] = max( abs( A(i,0:i) )), i=0:n-1, for ||A||_max, where A is stored upper.
 * TODO compare performance with lower case and use that implementation if better. */
__global__ void
zlanhe_max_kernel_u(
    int n, const magmaDoubleComplex* A, int lda, double *dwork )
{
    int ind = blockIdx.x * max_bs + threadIdx.x;
    double res = 0.;

    A += ind;
    if (ind < n) {
        for(int j=n-1; j >= ind; j--)
            res = fmax( res, cuCabs( A[j*lda] ) );
        
        dwork[ind] = res;
    }
}


/* Computes dwork[i] = max( abs( A(i,:) )), i=0:n-1, for ||A||_max */
extern "C" void
zlanhe_max(
    magma_uplo_t uplo, int n, const magmaDoubleComplex *A, int lda, double *dwork )
{
    int blocks = (n - 1)/max_bs + 1;
    dim3 grid(blocks, 1, 1);
    dim3 threads(max_bs, 1, 1);

    if ( uplo == 'L' || uplo == 'l' ) {
        zlanhe_max_kernel_l<<< grid, threads, 0, magma_stream >>>
            ( n, A, lda, dwork );
    }
    else {
        zlanhe_max_kernel_u<<< grid, threads, 0, magma_stream >>>
            ( n, A, lda, dwork );
    }
}


/* ====================================================================== */
/*
    Purpose
    =======
    
    ZLANHE returns the value of the one norm, or the Frobenius norm, or
    the infinity norm, or the element of largest absolute value of a
    complex Hermitian matrix A.
    
       ZLANHE = ( max(abs(A(i,j))), NORM = 'M' or 'm'
                (
                ( norm1(A),         NORM = '1', 'O' or 'o'      ** supported only for CUDA_ARCH >= 200
                (
                ( normI(A),         NORM = 'I' or 'i'           ** supported only for CUDA_ARCH >= 200
                (
                ( normF(A),         NORM = 'F', 'f', 'E' or 'e' ** not yet supported
    
    where norm1 denotes the one norm of a matrix (maximum column sum),
    normI denotes the infinity norm of a matrix (maximum row sum) and
    normF denotes the Frobenius norm of a matrix (square root of sum of squares).
    Note that max(abs(A(i,j))) is not a consistent matrix norm.
    
    Returns ZLANHE < 0: if ZLANHE = -i, the i-th argument had an illegal value.
    
    Arguments:
    ==========
    
    NORM    (input) CHARACTER*1
            Specifies the value to be returned in ZLANHE as described above.
    
    UPLO    (input) CHARACTER*1
            Specifies whether the upper or lower triangular part of the
            Hermitian matrix A is to be referenced.
            = 'U': Upper triangular part of A is referenced
            = 'L': Lower triangular part of A is referenced
    
    N       (input) INTEGER
            The order of the matrix A. N >= 0. When N = 0, ZLANHE is
            set to zero.
    
    A       (input) COMPLEX*16 array on the GPU, dimension (LDA,N)
            The Hermitian matrix A. If UPLO = 'U', the leading n by n
            upper triangular part of A contains the upper triangular part
            of the matrix A, and the strictly lower triangular part of A
            is not referenced. If UPLO = 'L', the leading n by n lower
            triangular part of A contains the lower triangular part of
            the matrix A, and the strictly upper triangular part of A is
            not referenced. Note that the imaginary parts of the diagonal
            elements need not be set and are assumed to be zero.
    
    LDA     (input) INTEGER
            The leading dimension of the array A. LDA >= max(N,1).
    
    DWORK   (workspace) DOUBLE PRECISION array on the GPU, dimension (MAX(1,LWORK)),
            where LWORK >= N.
            NOTE: this is different than LAPACK, where WORK is only required
            for norm1 and normI.
*/

extern "C" double
magmablas_zlanhe(
    magma_norm_t norm, magma_uplo_t uplo, magma_int_t n,
    const magmaDoubleComplex *A, magma_int_t lda, double *dwork )
{
    magma_int_t info = 0;
    magma_int_t arch = magma_getdevice_arch();
    // 1-norm == inf-norm since A is Hermitian
    bool inf_norm = (norm == 'I' || norm == 'i' || norm == '1' || norm == 'O' || norm == 'o');
    bool max_norm = (norm == 'M' || norm == 'm');
    if ( ! max_norm && (! inf_norm || arch < 200) )
        info = -1;
    else if ( uplo != 'u' && uplo != 'U' && uplo != 'l' && uplo != 'L' )
        info = -2;
    else if ( n < 0 )
        info = -3;
    else if ( lda < n )
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
        zlanhe_inf( uplo, n, A, lda, dwork );
        int i = cublasIdamax( n, dwork, 1 ) - 1;
        cudaMemcpy( &res, &dwork[i], sizeof(double), cudaMemcpyDeviceToHost );
    }
    else if ( max_norm ) {
        zlanhe_max( uplo, n, A, lda, dwork );
        int i = cublasIdamax( n, dwork, 1 ) - 1;
        cudaMemcpy( &res, &dwork[i], sizeof(double), cudaMemcpyDeviceToHost );
    }
    return res;
}
