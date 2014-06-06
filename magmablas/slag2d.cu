/*
    -- MAGMA (version 1.4.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       December 2013

       @generated ds Tue Dec 17 13:18:45 2013

*/
#include "common_magma.h"

__global__ void 
slag2d_generic(int M, int N, 
               const float *SA, int LDSA, 
               double       *A, int LDA ) 
{ 
    int ibx = blockIdx.x * 64;

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int idt = ty * 16 + tx;
        
    if( (ibx+idt) >= M ){
        SA += (M-1);
        A  += (M-1);
    }
    else{
        SA += ibx+idt;
        A  += ibx+idt;
    }
    const float * SAend = SA+LDSA*N;
    double Ap[1]={ (double)(SA[0]) };
    do {
        SA  += LDSA;
        A[0] = Ap[0];
        Ap[0]= (double)(SA[0]);
        A   += LDA;

    } while (SA < SAend);

    A[0] = Ap[0];
}

__global__ void 
slag2d_special(int M, int N, 
               const float *SA, int LDSA, 
               double       *A, int LDA ) 
{ 
    int ibx = blockIdx.x * 64;

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int idt = ty * 16 + tx;
        
    if( (ibx+idt) >= M ){
        SA += (M-1);
        A  += (M-1);
    }
    else{
        SA += ibx+idt;
        A  += ibx+idt;
    }
    double Ap[1] = { (double)(SA[0]) };
    A[0] = Ap[0];
}

extern "C" void 
magmablas_slag2d_64_64_16_4_v2( magma_int_t M, magma_int_t N, 
                                const float *SA, magma_int_t LDSA, 
                                double       *A, magma_int_t LDA )
{
    if( M == 0 || N==0 ) {
        printf("One of the dimension is ZERO\n");
        exit(-1);
    }
    dim3 threads( 16, 4 );
    dim3 grid(M/64+(M%64!=0),1);
    if( N > 1 ) {
        slag2d_generic<<< grid, threads, 0, magma_stream >>> (  M, N, SA, LDSA, A, LDA ) ;
    }
    else{
        slag2d_special<<< grid, threads, 0, magma_stream >>> (  M, N, SA, LDSA, A, LDA ) ;
    }
}

extern "C" void 
magmablas_slag2d(
    magma_int_t m, magma_int_t n,
    const float *SA, magma_int_t ldsa,
    double       *A, magma_int_t lda,
    magma_int_t *info)
{
/*
    Purpose
    =======
    
    SLAG2D converts a SINGLE PRECISION matrix, SA, to a DOUBLE
    PRECISION matrix, A.
    
    Note that while it is possible to overflow while converting
    from double to single, it is not possible to overflow when
    converting from single to double.
        
    Arguments
    =========
    
    M       (input) INTEGER
            The number of lines of the matrix A.  M >= 0.
    
    N       (input) INTEGER
            The number of columns of the matrix A.  N >= 0.
    
    SA      (input) REAL array, dimension (LDSA,N)
            On entry, the M-by-N coefficient matrix SA.
    
    LDSA    (input) INTEGER
            The leading dimension of the array SA.  LDSA >= max(1,M).
    
    A       (output) DOUBLE PRECISION array, dimension (LDA,N)
            On exit, the M-by-N coefficient matrix A.
    
    LDA     (input) INTEGER
            The leading dimension of the array A.  LDA >= max(1,M).
    
    INFO    (output) INTEGER
            = 0:  successful exit
            < 0:  if INFO = -i, the i-th argument had an illegal value
    =====================================================================    */
    
    *info = 0;
    if ( m < 0 )
        *info = -1;
    else if ( n < 0 )
        *info = -2;
    else if ( ldsa < max(1,m) )
        *info = -4;
    else if ( lda < max(1,m) )
        *info = -6;
    
    if (*info != 0) {
        magma_xerbla( __func__, -(*info) );
        //return *info;
    }
    
    magmablas_slag2d_64_64_16_4_v2( m, n, SA, ldsa, A, lda );
}        
