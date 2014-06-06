
/*
    -- MAGMA (version 1.5.0-beta2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date May 2014

       @precisions normal z -> c d s

       @author Peng Du
       @author Tingxing Dong
       
       File named ztrtri_kernel.cu to avoid name conflict with src/ztrtri.o
       in the library.
*/

#include "common_magma.h"

#define qmod(a, b) ((a)-(__mul24((b), (a)/(b))))

#define BLOCK_SIZE 16 // inner blocking size, <=32
#define NB 128        // outer blocking size, >BLOCK_SIZE

__global__ void
diag_ztrtri_kernel_upper(magma_diag_t diag, const magmaDoubleComplex *A, magmaDoubleComplex *d_dinvA, int lda)
{
    int i, j;
    magmaDoubleComplex Ystx = MAGMA_Z_ZERO;
    magmaDoubleComplex *y = NULL;
    magmaDoubleComplex switcher;
    magmaDoubleComplex neg_switcher;

    // Thread index
    int tx = threadIdx.x;

    // Block index
    int bx = blockIdx.x;

    const magmaDoubleComplex *Aoff = A + bx*lda*BLOCK_SIZE + bx*BLOCK_SIZE;
    int NumBLperNB = NB/BLOCK_SIZE;
    d_dinvA += bx/NumBLperNB*NB*NB + (bx % NumBLperNB)*(NB*BLOCK_SIZE + BLOCK_SIZE);

    __shared__ magmaDoubleComplex Bs[BLOCK_SIZE*BLOCK_SIZE];
    __shared__ magmaDoubleComplex workspace[BLOCK_SIZE];    // workspace used to store the current working column

    // load A
    #pragma unroll
    for( i=0; i < BLOCK_SIZE; i++ )
    {
        if(tx <= i)
        {
           Bs[i*BLOCK_SIZE+tx] = *(Aoff+i*lda+tx);    
        }
        else
        {
           Bs[i*BLOCK_SIZE+tx] = MAGMA_Z_ZERO; 
        }
    } 
// read in the whole square block of my A and zero out the non data triangular
 
    // Synchronize to make sure the matrices are loaded
    __syncthreads();

   // solve the diagonals

    if(diag == MagmaUnit)
    {
      Bs[tx*BLOCK_SIZE+tx] = MAGMA_Z_ONE;
    }
    else
    {
      if( Bs[tx*BLOCK_SIZE+tx] == MAGMA_Z_ZERO )
      {
         Bs[tx*BLOCK_SIZE+tx] = MAGMA_Z_ONE;  
      }
      else
      {
         Bs[tx*BLOCK_SIZE+tx] = MAGMA_Z_ONE / ( Bs[tx*BLOCK_SIZE+tx]) ;
      }      
    }


    /* the upper case */
    for( i=0; i < BLOCK_SIZE; i++ ) {
        Ystx =  MAGMA_Z_ZERO;
        if( tx < i)
        {
          switcher = MAGMA_Z_ONE;
        }
        else
        {
          switcher = MAGMA_Z_ZERO;
        }

        //dtrmv
        workspace[tx] = *(Bs+i*BLOCK_SIZE+tx);
        y = Bs+i*BLOCK_SIZE;

        #pragma unroll
        //for( j=tx; j < i; j++ )
        for( j=0; j < i; j++ )
            Ystx += switcher * (*(Bs+j*BLOCK_SIZE+tx)*workspace[j]);

        //sscal
        // if (tx != i) y[tx]=switcher*Ystx*(-Bs[i*BLOCK_SIZE+i]);

        if( tx != i)
        {
           switcher = MAGMA_Z_ONE;
           neg_switcher =  MAGMA_Z_ZERO;
        }
        else
        {
          switcher = MAGMA_Z_ZERO;
          neg_switcher =  MAGMA_Z_ONE;
        }

        y[tx] = switcher *Ystx*(-Bs[i*BLOCK_SIZE+i])+neg_switcher*y[tx];

        __syncthreads();
    }

    // write back A
    #pragma unroll
    for( i=0; i < BLOCK_SIZE; i++ )
        *(d_dinvA+i*NB+tx) = Bs[i*BLOCK_SIZE+tx];
}

__global__ void
diag_ztrtri_kernel_lower(magma_diag_t diag, const magmaDoubleComplex *A, magmaDoubleComplex *d_dinvA, int lda)
{
    int i, j;
    magmaDoubleComplex Ystx=  MAGMA_Z_ZERO;
    magmaDoubleComplex *Bw=NULL, *x=NULL, *y=NULL;
    magmaDoubleComplex switcher;
    magmaDoubleComplex neg_switcher;


    // Thread index
    int tx = threadIdx.x;
    int txw;

    // Block index
    int bx = blockIdx.x;

    const magmaDoubleComplex *Aoff = A+bx*lda*BLOCK_SIZE+bx*BLOCK_SIZE;
    int NumBLperNB = NB/BLOCK_SIZE;
    d_dinvA += bx/NumBLperNB*NB*NB + (bx % NumBLperNB)*(NB*BLOCK_SIZE + BLOCK_SIZE);

    __shared__ magmaDoubleComplex Bs[BLOCK_SIZE*BLOCK_SIZE];
    __shared__ magmaDoubleComplex workspace[BLOCK_SIZE];    // workspace used to store the current working column

    // load A
    #pragma unroll
    for( i=0; i < BLOCK_SIZE; i++ )
    {
        if(tx >= i)
        {
           Bs[i*BLOCK_SIZE+tx] = *(Aoff+i*lda+tx);
        }
        else
        {
           Bs[i*BLOCK_SIZE+tx] = MAGMA_Z_ZERO;
        }
    }

    // read in the whole square block of my A and zero out the non data triangular
    // not the upper or lower diagonal

    // Synchronize to make sure the matrices are loaded
    __syncthreads();

   // solve the diagonals

    if(diag == MagmaUnit)
    {
      Bs[tx*BLOCK_SIZE+tx] = MAGMA_Z_ONE;
    }
    else
    {
      if( Bs[tx*BLOCK_SIZE+tx] == MAGMA_Z_ZERO )
      {
         Bs[tx*BLOCK_SIZE+tx] = MAGMA_Z_ONE;  
      }
      else
      {
         Bs[tx*BLOCK_SIZE+tx] = MAGMA_Z_ONE / ( Bs[tx*BLOCK_SIZE+tx]) ;
      }      
    }

    /*
     * the lower case
     */


    if( !(tx < BLOCK_SIZE-1) )
    {
       switcher = MAGMA_Z_ONE;
    }
    else
    {
       switcher = MAGMA_Z_ZERO;
    }

    Bs[(BLOCK_SIZE-1)*BLOCK_SIZE+tx] = switcher * Bs[(BLOCK_SIZE-1)*BLOCK_SIZE+tx];    
   // zero out the last column, except the diagonal element

    for( i=BLOCK_SIZE-2; i >= 0; i-- ) {
        Ystx =  MAGMA_Z_ZERO;
       
        if( tx > i)
        {
          switcher = MAGMA_Z_ONE;
        }
        else
        {
          switcher = MAGMA_Z_ZERO;
        }

        //dtrmv
        Bw = Bs+(i+1)*BLOCK_SIZE+i+1;
        workspace[tx] = *(Bs+i*BLOCK_SIZE+tx);
        x = workspace+i+1;
        y = Bs+i*BLOCK_SIZE;

        txw = (tx-i-1);

        #pragma unroll
        for( j=0; j < BLOCK_SIZE-i-1; j++ )
            Ystx += switcher*(*(Bw+j*BLOCK_SIZE+txw)*x[j]);

        //sscal

        if( tx != i)
        {
           switcher = MAGMA_Z_ONE;
           neg_switcher =  MAGMA_Z_ZERO;
        }
        else
        {
          switcher = MAGMA_Z_ZERO;
          neg_switcher =  MAGMA_Z_ONE;
        }

        y[tx] = switcher * Ystx*(-Bs[i*BLOCK_SIZE+i])+ neg_switcher *y[tx];

        __syncthreads();
    }

    // write back A
    #pragma unroll
    for( i=0; i < BLOCK_SIZE; i++ )
        *(d_dinvA+i*NB+tx) = Bs[i*BLOCK_SIZE+tx];
}

/*
 * daxpy computes c += alpha*b, where b and c are 16-element vectors.
 */
static __device__ void daxpy(
    magmaDoubleComplex alpha,
    const magmaDoubleComplex * __restrict__ b,
    magmaDoubleComplex       * __restrict__ c )
{
    c[0]  += alpha * b[0];
    c[1]  += alpha * b[1];
    c[2]  += alpha * b[2];
    c[3]  += alpha * b[3];
    c[4]  += alpha * b[4];
    c[5]  += alpha * b[5];
    c[6]  += alpha * b[6];
    c[7]  += alpha * b[7];
    c[8]  += alpha * b[8];
    c[9]  += alpha * b[9];
    c[10] += alpha * b[10];
    c[11] += alpha * b[11];
    c[12] += alpha * b[12];
    c[13] += alpha * b[13];
    c[14] += alpha * b[14];
    c[15] += alpha * b[15];
}

__device__ void zgemm_kernel_16(
    magmaDoubleComplex *A, int lda,
    magmaDoubleComplex *B, int ldb,
    magmaDoubleComplex *C, int ldc,
    magmaDoubleComplex alpha, int blk, int inx, int iny, magmaDoubleComplex *c)
{
    const magmaDoubleComplex *Blast = B + blk;
    __shared__ magmaDoubleComplex bs[16][17];

    do {
        magmaDoubleComplex a[4] = { A[0*lda], A[1*lda], A[2*lda], A[3*lda] };

        bs[inx   ][iny   ] = B[    0*ldb];
        bs[inx   ][iny+ 4] = B[    4*ldb];
        bs[inx   ][iny+ 8] = B[    8*ldb];
        bs[inx   ][iny+12] = B[   12*ldb];
        bs[inx+ 4][iny   ] = B[ 4+ 0*ldb];
        bs[inx+ 4][iny+ 4] = B[ 4+ 4*ldb];
        bs[inx+ 4][iny+ 8] = B[ 4+ 8*ldb];
        bs[inx+ 4][iny+12] = B[ 4+12*ldb];
        bs[inx+ 8][iny   ] = B[ 8+ 0*ldb];
        bs[inx+ 8][iny+ 4] = B[ 8+ 4*ldb];
        bs[inx+ 8][iny+ 8] = B[ 8+ 8*ldb];
        bs[inx+ 8][iny+12] = B[ 8+12*ldb];
        bs[inx+12][iny   ] = B[12+ 0*ldb];
        bs[inx+12][iny+ 4] = B[12+ 4*ldb];
        bs[inx+12][iny+ 8] = B[12+ 8*ldb];
        bs[inx+12][iny+12] = B[12+12*ldb];
        __syncthreads();

        A += 4*lda;
        daxpy( a[0], &bs[ 0][0], c );  a[0] = A[0*lda];
        daxpy( a[1], &bs[ 1][0], c );  a[1] = A[1*lda];
        daxpy( a[2], &bs[ 2][0], c );  a[2] = A[2*lda];
        daxpy( a[3], &bs[ 3][0], c );  a[3] = A[3*lda];

        A += 4*lda;
        daxpy( a[0], &bs[ 4][0], c );  a[0] = A[0*lda];
        daxpy( a[1], &bs[ 5][0], c );  a[1] = A[1*lda];
        daxpy( a[2], &bs[ 6][0], c );  a[2] = A[2*lda];
        daxpy( a[3], &bs[ 7][0], c );  a[3] = A[3*lda];

        A += 4*lda;
        daxpy( a[0], &bs[ 8][0], c );  a[0] = A[0*lda];
        daxpy( a[1], &bs[ 9][0], c );  a[1] = A[1*lda];
        daxpy( a[2], &bs[10][0], c );  a[2] = A[2*lda];
        daxpy( a[3], &bs[11][0], c );  a[3] = A[3*lda];

        A += 4*lda;
        daxpy( a[0], &bs[12][0], c );
        daxpy( a[1], &bs[13][0], c );
        daxpy( a[2], &bs[14][0], c );
        daxpy( a[3], &bs[15][0], c );

        B += 16;
        __syncthreads();
    } while( B < Blast );

    for( int i = 0; i < 16; i++ ) {
        C[0] = alpha*c[i];
        C += ldc;
    }
}

/*
 * B21 = -inv(A11)*A12*inv(A22)
 */
__global__ void
triple_zgemm_update_16_R (const magmaDoubleComplex *Ain, magmaDoubleComplex *d_dinvA, int blk, int lda, int npages)
{
    const int bIdy = blockIdx.y/npages;
    //const int page = (blockIdx.y)%(npages);
    const int page = qmod(blockIdx.y, npages);
    const int inx = threadIdx.x;
    const int iny = threadIdx.y;
    const int ibx = blockIdx.x * (blockDim.x*blockDim.y);
    const int iby = bIdy * 16;
    const int id = inx + iny*blockDim.x;
    __shared__ magmaDoubleComplex bs[16][17];

    int PagesPerNB = NB/(blk*2);
    //--------------------------part one---------------------------//
    {
        // A12*inv(A22) -> A12
        // A=A12, B=inv(A22), C=A12(d_dinvA)
        const magmaDoubleComplex *A;
        magmaDoubleComplex *B, *C;
        int ldb = NB;
        int ldc = NB;

        d_dinvA += NB*NB*(page/PagesPerNB)
                + (qmod(page, PagesPerNB))*(blk*2)*NB
                + (qmod(page, PagesPerNB))*(blk*2);

        A = Ain + page*lda*blk*2 + blk*lda + page*blk*2;
        B = d_dinvA + blk*NB + blk;
        C = d_dinvA + blk*NB;

        A += ibx + id;
        B += inx + __mul24( iby + iny, ldb );
        C += ibx + id  + __mul24( iby, ldc );

        const magmaDoubleComplex *Blast = B + blk;

        magmaDoubleComplex c[16] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

        do {
            magmaDoubleComplex a[4] = { A[0*lda], A[1*lda], A[2*lda], A[3*lda] };

            bs[inx   ][iny   ] = B[    0*ldb];
            bs[inx   ][iny+ 4] = B[    4*ldb];
            bs[inx   ][iny+ 8] = B[    8*ldb];
            bs[inx   ][iny+12] = B[   12*ldb];
            bs[inx+ 4][iny   ] = B[ 4+ 0*ldb];
            bs[inx+ 4][iny+ 4] = B[ 4+ 4*ldb];
            bs[inx+ 4][iny+ 8] = B[ 4+ 8*ldb];
            bs[inx+ 4][iny+12] = B[ 4+12*ldb];
            bs[inx+ 8][iny   ] = B[ 8+ 0*ldb];
            bs[inx+ 8][iny+ 4] = B[ 8+ 4*ldb];
            bs[inx+ 8][iny+ 8] = B[ 8+ 8*ldb];
            bs[inx+ 8][iny+12] = B[ 8+12*ldb];
            bs[inx+12][iny   ] = B[12+ 0*ldb];
            bs[inx+12][iny+ 4] = B[12+ 4*ldb];
            bs[inx+12][iny+ 8] = B[12+ 8*ldb];
            bs[inx+12][iny+12] = B[12+12*ldb];
            __syncthreads();

            A += 4*lda;
            daxpy( a[0], &bs[ 0][0], c );  a[0] = A[0*lda];
            daxpy( a[1], &bs[ 1][0], c );  a[1] = A[1*lda];
            daxpy( a[2], &bs[ 2][0], c );  a[2] = A[2*lda];
            daxpy( a[3], &bs[ 3][0], c );  a[3] = A[3*lda];

            A += 4*lda;
            daxpy( a[0], &bs[ 4][0], c );  a[0] = A[0*lda];
            daxpy( a[1], &bs[ 5][0], c );  a[1] = A[1*lda];
            daxpy( a[2], &bs[ 6][0], c );  a[2] = A[2*lda];
            daxpy( a[3], &bs[ 7][0], c );  a[3] = A[3*lda];

            A += 4*lda;
            daxpy( a[0], &bs[ 8][0], c );  a[0] = A[0*lda];
            daxpy( a[1], &bs[ 9][0], c );  a[1] = A[1*lda];
            daxpy( a[2], &bs[10][0], c );  a[2] = A[2*lda];
            daxpy( a[3], &bs[11][0], c );  a[3] = A[3*lda];

            A += 4*lda;
            daxpy( a[0], &bs[12][0], c );
            daxpy( a[1], &bs[13][0], c );
            daxpy( a[2], &bs[14][0], c );
            daxpy( a[3], &bs[15][0], c );

            B += 16;
            __syncthreads();
        } while( B < Blast );

        for( int i = 0; i < 16; i++ ) {
            C[0] = c[i];
            C += ldc;
        }
    }
    __syncthreads();

    //--------------------------part two---------------------------//
    {
        // -inv(A11)*A12 -> A12
        // A=inv(A11), B=A12, C=A12
        magmaDoubleComplex *A, *B, *C;
        int lda = NB;
        int ldb = NB;
        int ldc = NB;

        A = d_dinvA;
        B = C = d_dinvA + blk*NB;

        A += ibx + id;
        B += inx + __mul24( iby + iny, ldb );
        C += ibx + id  + __mul24( iby, ldc );

        const magmaDoubleComplex *Blast = B + blk;

        magmaDoubleComplex c[16] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

        do {
            magmaDoubleComplex a[4] = { A[0*lda], A[1*lda], A[2*lda], A[3*lda] };

            bs[inx   ][iny   ] = B[    0*ldb];
            bs[inx   ][iny+ 4] = B[    4*ldb];
            bs[inx   ][iny+ 8] = B[    8*ldb];
            bs[inx   ][iny+12] = B[   12*ldb];
            bs[inx+ 4][iny   ] = B[ 4+ 0*ldb];
            bs[inx+ 4][iny+ 4] = B[ 4+ 4*ldb];
            bs[inx+ 4][iny+ 8] = B[ 4+ 8*ldb];
            bs[inx+ 4][iny+12] = B[ 4+12*ldb];
            bs[inx+ 8][iny   ] = B[ 8+ 0*ldb];
            bs[inx+ 8][iny+ 4] = B[ 8+ 4*ldb];
            bs[inx+ 8][iny+ 8] = B[ 8+ 8*ldb];
            bs[inx+ 8][iny+12] = B[ 8+12*ldb];
            bs[inx+12][iny   ] = B[12+ 0*ldb];
            bs[inx+12][iny+ 4] = B[12+ 4*ldb];
            bs[inx+12][iny+ 8] = B[12+ 8*ldb];
            bs[inx+12][iny+12] = B[12+12*ldb];
            __syncthreads();

            A += 4*lda;
            daxpy( a[0], &bs[ 0][0], c );  a[0] = A[0*lda];
            daxpy( a[1], &bs[ 1][0], c );  a[1] = A[1*lda];
            daxpy( a[2], &bs[ 2][0], c );  a[2] = A[2*lda];
            daxpy( a[3], &bs[ 3][0], c );  a[3] = A[3*lda];

            A += 4*lda;
            daxpy( a[0], &bs[ 4][0], c );  a[0] = A[0*lda];
            daxpy( a[1], &bs[ 5][0], c );  a[1] = A[1*lda];
            daxpy( a[2], &bs[ 6][0], c );  a[2] = A[2*lda];
            daxpy( a[3], &bs[ 7][0], c );  a[3] = A[3*lda];

            A += 4*lda;
            daxpy( a[0], &bs[ 8][0], c );  a[0] = A[0*lda];
            daxpy( a[1], &bs[ 9][0], c );  a[1] = A[1*lda];
            daxpy( a[2], &bs[10][0], c );  a[2] = A[2*lda];
            daxpy( a[3], &bs[11][0], c );  a[3] = A[3*lda];

            A += 4*lda;
            daxpy( a[0], &bs[12][0], c );
            daxpy( a[1], &bs[13][0], c );
            daxpy( a[2], &bs[14][0], c );
            daxpy( a[3], &bs[15][0], c );

            B += 16;
            __syncthreads();
        } while( B < Blast );

        for( int i = 0; i < 16; i++ ) {
            C[0] = (-1)*c[i];
            C += ldc;
        }
    }
}

/*
 * B21 = -inv(A22)*A21*inv(A11)
 */
__global__ void
triple_zgemm_update_16_part1_L (const magmaDoubleComplex *Ain, magmaDoubleComplex *d_dinvA, int blk, int lda, int npages)
{
    const int bIdy = blockIdx.y/npages;
    //const int page = (blockIdx.y)%(npages);
    const int page = qmod(blockIdx.y, npages);
    const int inx = threadIdx.x;
    const int iny = threadIdx.y;
    const int ibx = blockIdx.x * (blockDim.x*blockDim.y);
    const int iby = bIdy * 16;
    const int id = inx + iny*blockDim.x;
    __shared__ magmaDoubleComplex bs[16][17];

    //--------------------------part one---------------------------//
    {
        // A21*inv(A11) -> A21
        // A=A21, B=inv(A11), C=A21
        const magmaDoubleComplex *A;
        magmaDoubleComplex *B, *C;
        int ldb = NB;
        int ldc = NB;

        int PagesPerNB = NB/(blk*2);

        d_dinvA += NB*NB*(page/PagesPerNB)
                + (qmod(page, PagesPerNB))*(blk*2)*NB
                + (qmod(page, PagesPerNB))*(blk*2);

        A = Ain + page*lda*blk*2 + page*blk*2 + blk;
        B = d_dinvA;
        C = d_dinvA + blk;

        A += ibx + id;
        B += inx + __mul24( iby + iny, ldb );
        C += ibx + id  + __mul24( iby, ldc );

        const magmaDoubleComplex *Blast = B + blk;

        magmaDoubleComplex c[16] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

        do {
            magmaDoubleComplex a[4] = { A[0*lda], A[1*lda], A[2*lda], A[3*lda] };

            bs[inx   ][iny   ] = B[    0*ldb];
            bs[inx   ][iny+ 4] = B[    4*ldb];
            bs[inx   ][iny+ 8] = B[    8*ldb];
            bs[inx   ][iny+12] = B[   12*ldb];
            bs[inx+ 4][iny   ] = B[ 4+ 0*ldb];
            bs[inx+ 4][iny+ 4] = B[ 4+ 4*ldb];
            bs[inx+ 4][iny+ 8] = B[ 4+ 8*ldb];
            bs[inx+ 4][iny+12] = B[ 4+12*ldb];
            bs[inx+ 8][iny   ] = B[ 8+ 0*ldb];
            bs[inx+ 8][iny+ 4] = B[ 8+ 4*ldb];
            bs[inx+ 8][iny+ 8] = B[ 8+ 8*ldb];
            bs[inx+ 8][iny+12] = B[ 8+12*ldb];
            bs[inx+12][iny   ] = B[12+ 0*ldb];
            bs[inx+12][iny+ 4] = B[12+ 4*ldb];
            bs[inx+12][iny+ 8] = B[12+ 8*ldb];
            bs[inx+12][iny+12] = B[12+12*ldb];
            __syncthreads();

            A += 4*lda;
            daxpy( a[0], &bs[ 0][0], c );  a[0] = A[0*lda];
            daxpy( a[1], &bs[ 1][0], c );  a[1] = A[1*lda];
            daxpy( a[2], &bs[ 2][0], c );  a[2] = A[2*lda];
            daxpy( a[3], &bs[ 3][0], c );  a[3] = A[3*lda];

            A += 4*lda;
            daxpy( a[0], &bs[ 4][0], c );  a[0] = A[0*lda];
            daxpy( a[1], &bs[ 5][0], c );  a[1] = A[1*lda];
            daxpy( a[2], &bs[ 6][0], c );  a[2] = A[2*lda];
            daxpy( a[3], &bs[ 7][0], c );  a[3] = A[3*lda];

            A += 4*lda;
            daxpy( a[0], &bs[ 8][0], c );  a[0] = A[0*lda];
            daxpy( a[1], &bs[ 9][0], c );  a[1] = A[1*lda];
            daxpy( a[2], &bs[10][0], c );  a[2] = A[2*lda];
            daxpy( a[3], &bs[11][0], c );  a[3] = A[3*lda];

            A += 4*lda;
            daxpy( a[0], &bs[12][0], c );
            daxpy( a[1], &bs[13][0], c );
            daxpy( a[2], &bs[14][0], c );
            daxpy( a[3], &bs[15][0], c );

            B += 16;
            __syncthreads();
        } while( B < Blast );

        for( int i = 0; i < 16; i++ ) {
            C[0] = c[i];
            C += ldc;
        }
    }

    __syncthreads();
}

/*
 * B21 = -inv(A22)*A21*inv(A11)
 */
__global__ void
triple_zgemm_update_16_part2_L (const magmaDoubleComplex *Ain, magmaDoubleComplex *d_dinvA, int blk, int lda, int npages)
{
    const int bIdy = blockIdx.y/npages;
    const int page = qmod(blockIdx.y, npages);
    const int inx = threadIdx.x;
    const int iny = threadIdx.y;
    const int ibx = blockIdx.x * (blockDim.x*blockDim.y);
    const int iby = bIdy * 16;
    const int id = inx + iny*blockDim.x;
    __shared__ magmaDoubleComplex bs[16][17];

    //--------------------------part two---------------------------//
    {
        // -inv(A22)*A21 -> A21
        // A=inv(A22), B=A21, C=A21
        magmaDoubleComplex *A, *B, *C;
        int lda = NB;
        int ldb = NB;
        int ldc = NB;

        int PagesPerNB = NB/(blk*2);
        d_dinvA += NB*NB*(page/PagesPerNB)
                + (qmod(page, PagesPerNB))*(blk*2)*NB
                + (qmod(page, PagesPerNB))*(blk*2);

        A = d_dinvA + blk*NB + blk;
        B = C = d_dinvA + blk;

        A += ibx + id;
        B += inx + __mul24( iby + iny, ldb );
        C += ibx + id  + __mul24( iby, ldc );

        const magmaDoubleComplex *Blast = B + blk;

        magmaDoubleComplex c[16] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

        do {
            magmaDoubleComplex a[4] = { A[0*lda], A[1*lda], A[2*lda], A[3*lda] };

            bs[inx   ][iny   ] = B[    0*ldb];
            bs[inx   ][iny+ 4] = B[    4*ldb];
            bs[inx   ][iny+ 8] = B[    8*ldb];
            bs[inx   ][iny+12] = B[   12*ldb];
            bs[inx+ 4][iny   ] = B[ 4+ 0*ldb];
            bs[inx+ 4][iny+ 4] = B[ 4+ 4*ldb];
            bs[inx+ 4][iny+ 8] = B[ 4+ 8*ldb];
            bs[inx+ 4][iny+12] = B[ 4+12*ldb];
            bs[inx+ 8][iny   ] = B[ 8+ 0*ldb];
            bs[inx+ 8][iny+ 4] = B[ 8+ 4*ldb];
            bs[inx+ 8][iny+ 8] = B[ 8+ 8*ldb];
            bs[inx+ 8][iny+12] = B[ 8+12*ldb];
            bs[inx+12][iny   ] = B[12+ 0*ldb];
            bs[inx+12][iny+ 4] = B[12+ 4*ldb];
            bs[inx+12][iny+ 8] = B[12+ 8*ldb];
            bs[inx+12][iny+12] = B[12+12*ldb];
            __syncthreads();

            A += 4*lda;
            daxpy( a[0], &bs[ 0][0], c );  a[0] = A[0*lda];
            daxpy( a[1], &bs[ 1][0], c );  a[1] = A[1*lda];
            daxpy( a[2], &bs[ 2][0], c );  a[2] = A[2*lda];
            daxpy( a[3], &bs[ 3][0], c );  a[3] = A[3*lda];

            A += 4*lda;
            daxpy( a[0], &bs[ 4][0], c );  a[0] = A[0*lda];
            daxpy( a[1], &bs[ 5][0], c );  a[1] = A[1*lda];
            daxpy( a[2], &bs[ 6][0], c );  a[2] = A[2*lda];
            daxpy( a[3], &bs[ 7][0], c );  a[3] = A[3*lda];

            A += 4*lda;
            daxpy( a[0], &bs[ 8][0], c );  a[0] = A[0*lda];
            daxpy( a[1], &bs[ 9][0], c );  a[1] = A[1*lda];
            daxpy( a[2], &bs[10][0], c );  a[2] = A[2*lda];
            daxpy( a[3], &bs[11][0], c );  a[3] = A[3*lda];

            A += 4*lda;
            daxpy( a[0], &bs[12][0], c );
            daxpy( a[1], &bs[13][0], c );
            daxpy( a[2], &bs[14][0], c );
            daxpy( a[3], &bs[15][0], c );

            B += 16;
            __syncthreads();
        } while( B < Blast );

        for( int i = 0; i < 16; i++ ) {
            C[0] = (-1)*c[i];
            C += ldc;
        }
    }
    __syncthreads();
}

/*
 * B21 = -inv(A11)*A12*inv(A22)
 */
__global__ void
triple_zgemm_update_32_part1_R (const magmaDoubleComplex *Ain, magmaDoubleComplex *d_dinvA, int blk, int lda, int npages)
{
    const int bIdy = blockIdx.y/npages;
    const int page = qmod(blockIdx.y, npages);
    const int inx = threadIdx.x;
    const int iny = threadIdx.y;
    const int ibx = blockIdx.x * (blockDim.x*blockDim.y);
    const int iby = bIdy * 16;
    const int id = inx + iny*blockDim.x;
    __shared__ magmaDoubleComplex bs[16][17];

    int PagesPerNB = NB/(blk*2);
    //--------------------------part one---------------------------//
    {
        // A12*inv(A22) -> A21
        // A=A12, B=inv(A22), C=A12(d_dinvA)
        const magmaDoubleComplex *A;
        magmaDoubleComplex *B, *C;
        int ldb = NB;
        int ldc = NB;

        d_dinvA += NB*NB*(page/PagesPerNB)
                + (qmod(page, PagesPerNB))*(blk*2)*NB
                + (qmod(page, PagesPerNB))*(blk*2);

        A = Ain + page*lda*blk*2 + blk*lda + page*blk*2;
        B = d_dinvA + blk*NB + blk;
        C = d_dinvA + blk*NB;

        A += ibx + id;
        B += inx + __mul24( iby + iny, ldb );
        C += ibx + id  + __mul24( iby, ldc );

        const magmaDoubleComplex *Blast = B + blk;

        magmaDoubleComplex c[16] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

        do {
            magmaDoubleComplex a[4] = { A[0*lda], A[1*lda], A[2*lda], A[3*lda] };

            bs[inx  ][iny   ] = B[   0*ldb];
            bs[inx  ][iny+ 4] = B[   4*ldb];
            bs[inx  ][iny+ 8] = B[   8*ldb];
            bs[inx  ][iny+12] = B[  12*ldb];
            bs[inx+8][iny   ] = B[8+ 0*ldb];
            bs[inx+8][iny+ 4] = B[8+ 4*ldb];
            bs[inx+8][iny+ 8] = B[8+ 8*ldb];
            bs[inx+8][iny+12] = B[8+12*ldb];
            __syncthreads();

            A += 4*lda;
            daxpy( a[0], &bs[ 0][0], c );  a[0] = A[0*lda];
            daxpy( a[1], &bs[ 1][0], c );  a[1] = A[1*lda];
            daxpy( a[2], &bs[ 2][0], c );  a[2] = A[2*lda];
            daxpy( a[3], &bs[ 3][0], c );  a[3] = A[3*lda];

            A += 4*lda;
            daxpy( a[0], &bs[ 4][0], c );  a[0] = A[0*lda];
            daxpy( a[1], &bs[ 5][0], c );  a[1] = A[1*lda];
            daxpy( a[2], &bs[ 6][0], c );  a[2] = A[2*lda];
            daxpy( a[3], &bs[ 7][0], c );  a[3] = A[3*lda];

            A += 4*lda;
            daxpy( a[0], &bs[ 8][0], c );  a[0] = A[0*lda];
            daxpy( a[1], &bs[ 9][0], c );  a[1] = A[1*lda];
            daxpy( a[2], &bs[10][0], c );  a[2] = A[2*lda];
            daxpy( a[3], &bs[11][0], c );  a[3] = A[3*lda];

            A += 4*lda;
            daxpy( a[0], &bs[12][0], c );
            daxpy( a[1], &bs[13][0], c );
            daxpy( a[2], &bs[14][0], c );
            daxpy( a[3], &bs[15][0], c );

            B += 16;
            __syncthreads();
        } while( B < Blast );

        for( int i = 0; i < 16; i++ ) {
            C[0] = c[i];
            C += ldc;
        }
    }

    __syncthreads();
}

/*
 * B21 = -inv(A11)*A12*inv(A22)
 */
__global__ void
triple_zgemm_update_32_part2_R (const magmaDoubleComplex *Ain, magmaDoubleComplex *d_dinvA, int blk, int lda, int npages)
{
    const int bIdy = blockIdx.y/npages;
    const int page = qmod(blockIdx.y, npages);
    const int inx = threadIdx.x;
    const int iny = threadIdx.y;
    const int ibx = blockIdx.x * (blockDim.x*blockDim.y);
    const int iby = bIdy * 16;
    const int id = inx + iny*blockDim.x;
    __shared__ magmaDoubleComplex bs[16][17];

    int PagesPerNB = NB/(blk*2);

    //--------------------------part two---------------------------//
    {
        // -inv(A11)*A12 -> A12
        // A=inv(A11), B=A12, C=A12
        magmaDoubleComplex *A, *B, *C;
        int lda = NB;
        int ldb = NB;
        int ldc = NB;

        d_dinvA += NB*NB*(page/PagesPerNB)
                + (qmod(page, PagesPerNB))*(blk*2)*NB
                + (qmod(page, PagesPerNB))*(blk*2);

        A = d_dinvA;
        B = C = d_dinvA + blk*NB;

        A += ibx + id;
        B += inx + __mul24( iby + iny, ldb );
        C += ibx + id  + __mul24( iby, ldc );

        const magmaDoubleComplex *Blast = B + blk;

        magmaDoubleComplex c[16] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

        do {
            magmaDoubleComplex a[4] = { A[0*lda], A[1*lda], A[2*lda], A[3*lda] };

            bs[inx  ][iny   ] = B[   0*ldb];
            bs[inx  ][iny+ 4] = B[   4*ldb];
            bs[inx  ][iny+ 8] = B[   8*ldb];
            bs[inx  ][iny+12] = B[  12*ldb];
            bs[inx+8][iny   ] = B[8+ 0*ldb];
            bs[inx+8][iny+ 4] = B[8+ 4*ldb];
            bs[inx+8][iny+ 8] = B[8+ 8*ldb];
            bs[inx+8][iny+12] = B[8+12*ldb];
            __syncthreads();

            A += 4*lda;
            daxpy( a[0], &bs[ 0][0], c );  a[0] = A[0*lda];
            daxpy( a[1], &bs[ 1][0], c );  a[1] = A[1*lda];
            daxpy( a[2], &bs[ 2][0], c );  a[2] = A[2*lda];
            daxpy( a[3], &bs[ 3][0], c );  a[3] = A[3*lda];

            A += 4*lda;
            daxpy( a[0], &bs[ 4][0], c );  a[0] = A[0*lda];
            daxpy( a[1], &bs[ 5][0], c );  a[1] = A[1*lda];
            daxpy( a[2], &bs[ 6][0], c );  a[2] = A[2*lda];
            daxpy( a[3], &bs[ 7][0], c );  a[3] = A[3*lda];

            A += 4*lda;
            daxpy( a[0], &bs[ 8][0], c );  a[0] = A[0*lda];
            daxpy( a[1], &bs[ 9][0], c );  a[1] = A[1*lda];
            daxpy( a[2], &bs[10][0], c );  a[2] = A[2*lda];
            daxpy( a[3], &bs[11][0], c );  a[3] = A[3*lda];

            A += 4*lda;
            daxpy( a[0], &bs[12][0], c );
            daxpy( a[1], &bs[13][0], c );
            daxpy( a[2], &bs[14][0], c );
            daxpy( a[3], &bs[15][0], c );

            B += 16;
            __syncthreads();
        } while( B < Blast );

        for( int i = 0; i < 16; i++ ) {
            C[0] = (-1)*c[i];
            C += ldc;
        }
    }
}

/*
 * B21 = -inv(A22)*A21*inv(A11)
 */
__global__ void
triple_zgemm_update_32_part1_L (const magmaDoubleComplex *Ain, magmaDoubleComplex *d_dinvA, int blk, int lda, int npages)
{
    const int bIdy = blockIdx.y/npages;
    const int page = qmod(blockIdx.y, npages);
    const int inx = threadIdx.x;
    const int iny = threadIdx.y;
    const int ibx = blockIdx.x * (blockDim.x*blockDim.y);
    const int iby = bIdy * 16;
    const int id = inx + iny*blockDim.x;
    __shared__ magmaDoubleComplex bs[16][17];

    int PagesPerNB = NB/(blk*2);
    //--------------------------part one---------------------------//
    {
        // A21*inv(A11) -> A21
        // A=A21, B=inv(A11), C=A21
        const magmaDoubleComplex *A;
        magmaDoubleComplex *B, *C;
        int ldb = NB;
        int ldc = NB;

        d_dinvA += NB*NB*(page/PagesPerNB)
                + (qmod(page, PagesPerNB))*(blk*2)*NB
                + (qmod(page, PagesPerNB))*(blk*2);

        A = Ain + page*lda*blk*2 + page*blk*2 + blk;
        B = d_dinvA;
        C = d_dinvA + blk;

        A += ibx + id;
        B += inx + __mul24( iby + iny, ldb );
        C += ibx + id  + __mul24( iby, ldc );

        const magmaDoubleComplex *Blast = B + blk;

        magmaDoubleComplex c[16] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

        do {
            magmaDoubleComplex a[4] = { A[0*lda], A[1*lda], A[2*lda], A[3*lda] };

            bs[inx  ][iny   ] = B[   0*ldb];
            bs[inx  ][iny+ 4] = B[   4*ldb];
            bs[inx  ][iny+ 8] = B[   8*ldb];
            bs[inx  ][iny+12] = B[  12*ldb];
            bs[inx+8][iny   ] = B[8+ 0*ldb];
            bs[inx+8][iny+ 4] = B[8+ 4*ldb];
            bs[inx+8][iny+ 8] = B[8+ 8*ldb];
            bs[inx+8][iny+12] = B[8+12*ldb];
            __syncthreads();

            A += 4*lda;
            daxpy( a[0], &bs[ 0][0], c );  a[0] = A[0*lda];
            daxpy( a[1], &bs[ 1][0], c );  a[1] = A[1*lda];
            daxpy( a[2], &bs[ 2][0], c );  a[2] = A[2*lda];
            daxpy( a[3], &bs[ 3][0], c );  a[3] = A[3*lda];

            A += 4*lda;
            daxpy( a[0], &bs[ 4][0], c );  a[0] = A[0*lda];
            daxpy( a[1], &bs[ 5][0], c );  a[1] = A[1*lda];
            daxpy( a[2], &bs[ 6][0], c );  a[2] = A[2*lda];
            daxpy( a[3], &bs[ 7][0], c );  a[3] = A[3*lda];

            A += 4*lda;
            daxpy( a[0], &bs[ 8][0], c );  a[0] = A[0*lda];
            daxpy( a[1], &bs[ 9][0], c );  a[1] = A[1*lda];
            daxpy( a[2], &bs[10][0], c );  a[2] = A[2*lda];
            daxpy( a[3], &bs[11][0], c );  a[3] = A[3*lda];

            A += 4*lda;
            daxpy( a[0], &bs[12][0], c );
            daxpy( a[1], &bs[13][0], c );
            daxpy( a[2], &bs[14][0], c );
            daxpy( a[3], &bs[15][0], c );

            B += 16;
            __syncthreads();
        } while( B < Blast );

        for( int i = 0; i < 16; i++ ) {
            C[0] = c[i];
            C += ldc;
        }
    }

    __syncthreads();
}

/*
 * B21 = -inv(A22)*A21*inv(A11)
 */
__global__ void
triple_zgemm_update_32_part2_L (const magmaDoubleComplex *Ain, magmaDoubleComplex *d_dinvA, int blk, int lda, int npages)
{
    const int bIdy = blockIdx.y/npages;
    const int page = qmod(blockIdx.y, npages);
    const int inx = threadIdx.x;
    const int iny = threadIdx.y;
    const int ibx = blockIdx.x * (blockDim.x*blockDim.y);
    const int iby = bIdy * 16;
    const int id = inx + iny*blockDim.x;
    __shared__ magmaDoubleComplex bs[16][17];

    int PagesPerNB = NB/(blk*2);
    //--------------------------part two---------------------------//
    {
        // -inv(A22)*A21 -> A21
        // A=inv(A22), B=A21, C=A21
        const magmaDoubleComplex *A;
        magmaDoubleComplex *B, *C;
        int lda = NB;
        int ldb = NB;
        int ldc = NB;

        d_dinvA += NB*NB*(page/PagesPerNB)
                + (qmod(page, PagesPerNB))*(blk*2)*NB
                + (qmod(page, PagesPerNB))*(blk*2);

        A = d_dinvA + blk*NB + blk;
        B = C = d_dinvA + blk;

        A += ibx + id;
        B += inx + __mul24( iby + iny, ldb );
        C += ibx + id  + __mul24( iby, ldc );

        const magmaDoubleComplex *Blast = B + blk;

        magmaDoubleComplex c[16] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

        do {
            magmaDoubleComplex a[4] = { A[0*lda], A[1*lda], A[2*lda], A[3*lda] };

            bs[inx  ][iny   ] = B[   0*ldb];
            bs[inx  ][iny+ 4] = B[   4*ldb];
            bs[inx  ][iny+ 8] = B[   8*ldb];
            bs[inx  ][iny+12] = B[  12*ldb];
            bs[inx+8][iny   ] = B[8+ 0*ldb];
            bs[inx+8][iny+ 4] = B[8+ 4*ldb];
            bs[inx+8][iny+ 8] = B[8+ 8*ldb];
            bs[inx+8][iny+12] = B[8+12*ldb];
            __syncthreads();

            A += 4*lda;
            daxpy( a[0], &bs[ 0][0], c );  a[0] = A[0*lda];
            daxpy( a[1], &bs[ 1][0], c );  a[1] = A[1*lda];
            daxpy( a[2], &bs[ 2][0], c );  a[2] = A[2*lda];
            daxpy( a[3], &bs[ 3][0], c );  a[3] = A[3*lda];

            A += 4*lda;
            daxpy( a[0], &bs[ 4][0], c );  a[0] = A[0*lda];
            daxpy( a[1], &bs[ 5][0], c );  a[1] = A[1*lda];
            daxpy( a[2], &bs[ 6][0], c );  a[2] = A[2*lda];
            daxpy( a[3], &bs[ 7][0], c );  a[3] = A[3*lda];

            A += 4*lda;
            daxpy( a[0], &bs[ 8][0], c );  a[0] = A[0*lda];
            daxpy( a[1], &bs[ 9][0], c );  a[1] = A[1*lda];
            daxpy( a[2], &bs[10][0], c );  a[2] = A[2*lda];
            daxpy( a[3], &bs[11][0], c );  a[3] = A[3*lda];

            A += 4*lda;
            daxpy( a[0], &bs[12][0], c );
            daxpy( a[1], &bs[13][0], c );
            daxpy( a[2], &bs[14][0], c );
            daxpy( a[3], &bs[15][0], c );

            B += 16;
            __syncthreads();
        } while( B < Blast );

        for( int i = 0; i < 16; i++ ) {
            C[0] = (-1)*c[i];
            C += ldc;
        }
    }
}

/*
 * B21 = -inv(A11)*A12*inv(A22)
 */
__global__ void
triple_zgemm_update_64_part1_R (const magmaDoubleComplex *Ain, magmaDoubleComplex *d_dinvA, int blk, int lda, int npages)
{
    const int bIdy = blockIdx.y/npages;
    const int page = qmod(blockIdx.y, npages);
    const int inx = threadIdx.x;
    const int iny = threadIdx.y;
    const int ibx = blockIdx.x*64;
    const int iby = bIdy*16;
    const int id = inx + iny*16;
    __shared__ magmaDoubleComplex bs[16][17];

    int PagesPerNB = NB/(blk*2);
    //--------------------------part one---------------------------//
    {
        // A12*inv(A22) -> A12(d_dinvA)
        // A=A12, B=inv(A22), C=A12
        const magmaDoubleComplex *A;
        magmaDoubleComplex *B, *C;
        int ldb = NB;
        int ldc = NB;

        d_dinvA += NB*NB*(page/PagesPerNB)
                + (qmod(page, PagesPerNB))*(blk*2)*NB
                + (qmod(page, PagesPerNB))*(blk*2);

        A = Ain + page*lda*blk*2 + blk*lda + page*blk*2;
        B = d_dinvA + blk*NB + blk;
        C = d_dinvA + blk*NB;

        A += ibx + id;
        B += inx + __mul24( iby + iny, ldb );
        C += ibx + id  + __mul24( iby, ldc );

        const magmaDoubleComplex *Blast = B + blk;

        magmaDoubleComplex c[16] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

        do {
            magmaDoubleComplex a[4] = { A[0*lda], A[1*lda], A[2*lda], A[3*lda] };

            bs[inx][iny   ] = B[ 0*ldb];
            bs[inx][iny+ 4] = B[ 4*ldb];
            bs[inx][iny+ 8] = B[ 8*ldb];
            bs[inx][iny+12] = B[12*ldb];
            __syncthreads();

            A += 4*lda;
            daxpy( a[0], &bs[ 0][0], c );  a[0] = A[0*lda];
            daxpy( a[1], &bs[ 1][0], c );  a[1] = A[1*lda];
            daxpy( a[2], &bs[ 2][0], c );  a[2] = A[2*lda];
            daxpy( a[3], &bs[ 3][0], c );  a[3] = A[3*lda];

            A += 4*lda;
            daxpy( a[0], &bs[ 4][0], c );  a[0] = A[0*lda];
            daxpy( a[1], &bs[ 5][0], c );  a[1] = A[1*lda];
            daxpy( a[2], &bs[ 6][0], c );  a[2] = A[2*lda];
            daxpy( a[3], &bs[ 7][0], c );  a[3] = A[3*lda];

            A += 4*lda;
            daxpy( a[0], &bs[ 8][0], c );  a[0] = A[0*lda];
            daxpy( a[1], &bs[ 9][0], c );  a[1] = A[1*lda];
            daxpy( a[2], &bs[10][0], c );  a[2] = A[2*lda];
            daxpy( a[3], &bs[11][0], c );  a[3] = A[3*lda];

            A += 4*lda;
            daxpy( a[0], &bs[12][0], c );
            daxpy( a[1], &bs[13][0], c );
            daxpy( a[2], &bs[14][0], c );
            daxpy( a[3], &bs[15][0], c );

            B += 16;
            __syncthreads();
        } while( B < Blast );

        for( int i = 0; i < 16; i++ ) {
            C[0] = c[i];
            C += ldc;
        }
    }
}

/*
 * B21 = -inv(A11)*A12*inv(A22)
 */
__global__ void
triple_zgemm_update_64_part2_R (const magmaDoubleComplex *Ain, magmaDoubleComplex *d_dinvA, int blk, int lda, int npages)
{
    const int bIdy = blockIdx.y/npages;
    const int page = qmod(blockIdx.y, npages);
    const int inx = threadIdx.x;
    const int iny = threadIdx.y;
    const int ibx = blockIdx.x*64;
    const int iby = bIdy*16;
    const int id = inx + iny*16;
    __shared__ magmaDoubleComplex bs[16][17];

    int PagesPerNB = NB/(blk*2);

    //--------------------------part two---------------------------//
    {
        // -inv(A11)*A12 -> A12
        // A=inv(A11), B=A12, C=A12
        const magmaDoubleComplex *A;
        magmaDoubleComplex *B, *C;
        int lda = NB;
        int ldb = NB;
        int ldc = NB;

        d_dinvA += NB*NB*(page/PagesPerNB)
                + (qmod(page, PagesPerNB))*(blk*2)*NB
                + (qmod(page, PagesPerNB))*(blk*2);

        A = d_dinvA;
        B = C = d_dinvA + blk*NB;

        A += ibx + id;
        B += inx + __mul24( iby + iny, ldb );
        C += ibx + id  + __mul24( iby, ldc );

        const magmaDoubleComplex *Blast = B + blk;

        magmaDoubleComplex c[16] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

        do {
            magmaDoubleComplex a[4] = { A[0*lda], A[1*lda], A[2*lda], A[3*lda] };

            bs[inx][iny   ] = B[ 0*ldb];
            bs[inx][iny+ 4] = B[ 4*ldb];
            bs[inx][iny+ 8] = B[ 8*ldb];
            bs[inx][iny+12] = B[12*ldb];
            __syncthreads();

            A += 4*lda;
            daxpy( a[0], &bs[ 0][0], c );  a[0] = A[0*lda];
            daxpy( a[1], &bs[ 1][0], c );  a[1] = A[1*lda];
            daxpy( a[2], &bs[ 2][0], c );  a[2] = A[2*lda];
            daxpy( a[3], &bs[ 3][0], c );  a[3] = A[3*lda];

            A += 4*lda;
            daxpy( a[0], &bs[ 4][0], c );  a[0] = A[0*lda];
            daxpy( a[1], &bs[ 5][0], c );  a[1] = A[1*lda];
            daxpy( a[2], &bs[ 6][0], c );  a[2] = A[2*lda];
            daxpy( a[3], &bs[ 7][0], c );  a[3] = A[3*lda];

            A += 4*lda;
            daxpy( a[0], &bs[ 8][0], c );  a[0] = A[0*lda];
            daxpy( a[1], &bs[ 9][0], c );  a[1] = A[1*lda];
            daxpy( a[2], &bs[10][0], c );  a[2] = A[2*lda];
            daxpy( a[3], &bs[11][0], c );  a[3] = A[3*lda];

            A += 4*lda;
            daxpy( a[0], &bs[12][0], c );
            daxpy( a[1], &bs[13][0], c );
            daxpy( a[2], &bs[14][0], c );
            daxpy( a[3], &bs[15][0], c );

            B += 16;
            __syncthreads();
        } while( B < Blast );

        for( int i = 0; i < 16; i++ ) {
            C[0] = (-1)*c[i];
            C += ldc;
        }
    }
}

/*
 * B21 = -inv(A22)*A21*inv(A11)
 */
__global__ void
triple_zgemm_update_64_part1_L (const magmaDoubleComplex *Ain, magmaDoubleComplex *d_dinvA, int blk, int lda, int npages)
{
    const int bIdy = blockIdx.y/npages;
    const int page = qmod(blockIdx.y, npages);
    const int inx = threadIdx.x;
    const int iny = threadIdx.y;
    const int ibx = blockIdx.x*64;
    const int iby = bIdy*16;
    const int id = inx + iny*16;
    __shared__ magmaDoubleComplex bs[16][17];

    int PagesPerNB = NB/(blk*2);
    //--------------------------part one---------------------------//
    {
        // A21*inv(A11) -> A21
        // A=A21, B=inv(A11), C=A21
        const magmaDoubleComplex *A;
        magmaDoubleComplex *B, *C;
        int ldb = NB;
        int ldc = NB;

        d_dinvA += NB*NB*(page/PagesPerNB)
                + (qmod(page, PagesPerNB))*(blk*2)*NB
                + (qmod(page, PagesPerNB))*(blk*2);

        A = Ain + page*lda*blk*2 + page*blk*2 + blk;
        B = d_dinvA;
        C = d_dinvA + blk;

        A += ibx + id;
        B += inx + __mul24( iby + iny, ldb );
        C += ibx + id  + __mul24( iby, ldc );

        const magmaDoubleComplex *Blast = B + blk;

        magmaDoubleComplex c[16] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

        do {
            magmaDoubleComplex a[4] = { A[0*lda], A[1*lda], A[2*lda], A[3*lda] };

            bs[inx][iny   ] = B[ 0*ldb];
            bs[inx][iny+ 4] = B[ 4*ldb];
            bs[inx][iny+ 8] = B[ 8*ldb];
            bs[inx][iny+12] = B[12*ldb];
            __syncthreads();

            A += 4*lda;
            daxpy( a[0], &bs[ 0][0], c );  a[0] = A[0*lda];
            daxpy( a[1], &bs[ 1][0], c );  a[1] = A[1*lda];
            daxpy( a[2], &bs[ 2][0], c );  a[2] = A[2*lda];
            daxpy( a[3], &bs[ 3][0], c );  a[3] = A[3*lda];

            A += 4*lda;
            daxpy( a[0], &bs[ 4][0], c );  a[0] = A[0*lda];
            daxpy( a[1], &bs[ 5][0], c );  a[1] = A[1*lda];
            daxpy( a[2], &bs[ 6][0], c );  a[2] = A[2*lda];
            daxpy( a[3], &bs[ 7][0], c );  a[3] = A[3*lda];

            A += 4*lda;
            daxpy( a[0], &bs[ 8][0], c );  a[0] = A[0*lda];
            daxpy( a[1], &bs[ 9][0], c );  a[1] = A[1*lda];
            daxpy( a[2], &bs[10][0], c );  a[2] = A[2*lda];
            daxpy( a[3], &bs[11][0], c );  a[3] = A[3*lda];

            A += 4*lda;
            daxpy( a[0], &bs[12][0], c );
            daxpy( a[1], &bs[13][0], c );
            daxpy( a[2], &bs[14][0], c );
            daxpy( a[3], &bs[15][0], c );

            B += 16;
            __syncthreads();
        } while( B < Blast );

        for( int i = 0; i < 16; i++ ) {
            C[0] = c[i];
            C += ldc;
        }
    }
}

/*
 * B21 = -inv(A22)*A21*inv(A11)
 */
__global__ void
triple_zgemm_update_64_part2_L (const magmaDoubleComplex *Ain, magmaDoubleComplex *d_dinvA, int blk, int lda, int npages)
{
    const int bIdy = blockIdx.y/npages;
    const int page = qmod(blockIdx.y, npages);
    const int inx = threadIdx.x;
    const int iny = threadIdx.y;
    const int ibx = blockIdx.x*64;
    const int iby = bIdy*16;
    const int id = inx + iny*16;
    __shared__ magmaDoubleComplex bs[16][17];

    int PagesPerNB = NB/(blk*2);

    //--------------------------part two---------------------------//
    {
        // -inv(A22)*A21 -> A21
        // A=inv(A22), B=A21, C=A21
        const magmaDoubleComplex *A;
        magmaDoubleComplex *B, *C;
        int lda = NB;
        int ldb = NB;
        int ldc = NB;

        d_dinvA += NB*NB*(page/PagesPerNB)
                + (qmod(page, PagesPerNB))*(blk*2)*NB
                + (qmod(page, PagesPerNB))*(blk*2);

        A = d_dinvA + blk*NB + blk;
        B = C = d_dinvA + blk;

        A += ibx + id;
        B += inx + __mul24( iby + iny, ldb );
        C += ibx + id  + __mul24( iby, ldc );

        const magmaDoubleComplex *Blast = B + blk;

        magmaDoubleComplex c[16] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

        do {
            magmaDoubleComplex a[4] = { A[0*lda], A[1*lda], A[2*lda], A[3*lda] };

            bs[inx][iny   ] = B[ 0*ldb];
            bs[inx][iny+ 4] = B[ 4*ldb];
            bs[inx][iny+ 8] = B[ 8*ldb];
            bs[inx][iny+12] = B[12*ldb];
            __syncthreads();

            A += 4*lda;
            daxpy( a[0], &bs[ 0][0], c );  a[0] = A[0*lda];
            daxpy( a[1], &bs[ 1][0], c );  a[1] = A[1*lda];
            daxpy( a[2], &bs[ 2][0], c );  a[2] = A[2*lda];
            daxpy( a[3], &bs[ 3][0], c );  a[3] = A[3*lda];

            A += 4*lda;
            daxpy( a[0], &bs[ 4][0], c );  a[0] = A[0*lda];
            daxpy( a[1], &bs[ 5][0], c );  a[1] = A[1*lda];
            daxpy( a[2], &bs[ 6][0], c );  a[2] = A[2*lda];
            daxpy( a[3], &bs[ 7][0], c );  a[3] = A[3*lda];

            A += 4*lda;
            daxpy( a[0], &bs[ 8][0], c );  a[0] = A[0*lda];
            daxpy( a[1], &bs[ 9][0], c );  a[1] = A[1*lda];
            daxpy( a[2], &bs[10][0], c );  a[2] = A[2*lda];
            daxpy( a[3], &bs[11][0], c );  a[3] = A[3*lda];

            A += 4*lda;
            daxpy( a[0], &bs[12][0], c );
            daxpy( a[1], &bs[13][0], c );
            daxpy( a[2], &bs[14][0], c );
            daxpy( a[3], &bs[15][0], c );

            B += 16;
            __syncthreads();
        } while( B < Blast );

        for( int i = 0; i < 16; i++ ) {
            C[0] = (-1)*c[i];
            C += ldc;
        }
    }
}

/*
 * B21 = -inv(A11)*A12*inv(A22)
 */
__global__ void
triple_zgemm_update_above64_part1_R (const magmaDoubleComplex *Ain, magmaDoubleComplex *d_dinvA, int blk, int lda, int npages)
{
    const int bIdy = blockIdx.y/npages;
    const int page = qmod(blockIdx.y, npages);
    const int inx = threadIdx.x;
    const int iny = threadIdx.y;
    const int ibx = blockIdx.x*64;
    const int iby = bIdy*16;
    const int id = inx + iny*16;
    __shared__ magmaDoubleComplex bs[16][17];

    int PagesPerNB = NB/(blk*2);
    //--------------------------part one---------------------------//
    {
        // A12*inv(A22) -> A12(d_dinvA)
        // A=A12, B=inv(A22), C=A12
        const magmaDoubleComplex *A;
        magmaDoubleComplex *B, *C;
        int ldb = NB;
        int ldc = NB;

        d_dinvA += NB*NB*(page/PagesPerNB)
                + (qmod(page, PagesPerNB))*(blk*2)*NB
                + (qmod(page, PagesPerNB))*(blk*2);

        A = Ain + page*lda*blk*2 + blk*lda + page*blk*2;
        B = d_dinvA + blk*NB + blk;
        C = d_dinvA + blk*NB;

        A += ibx + id;
        B += inx + __mul24( iby + iny, ldb );
        C += ibx + id  + __mul24( iby, ldc );

        const magmaDoubleComplex *Blast = B + blk;

        magmaDoubleComplex c[16] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

        do {
            magmaDoubleComplex a[4] = { A[0*lda], A[1*lda], A[2*lda], A[3*lda] };

            bs[inx][iny   ] = B[ 0*ldb];
            bs[inx][iny+ 4] = B[ 4*ldb];
            bs[inx][iny+ 8] = B[ 8*ldb];
            bs[inx][iny+12] = B[12*ldb];
            __syncthreads();

            A += 4*lda;
            daxpy( a[0], &bs[ 0][0], c );  a[0] = A[0*lda];
            daxpy( a[1], &bs[ 1][0], c );  a[1] = A[1*lda];
            daxpy( a[2], &bs[ 2][0], c );  a[2] = A[2*lda];
            daxpy( a[3], &bs[ 3][0], c );  a[3] = A[3*lda];

            A += 4*lda;
            daxpy( a[0], &bs[ 4][0], c );  a[0] = A[0*lda];
            daxpy( a[1], &bs[ 5][0], c );  a[1] = A[1*lda];
            daxpy( a[2], &bs[ 6][0], c );  a[2] = A[2*lda];
            daxpy( a[3], &bs[ 7][0], c );  a[3] = A[3*lda];

            A += 4*lda;
            daxpy( a[0], &bs[ 8][0], c );  a[0] = A[0*lda];
            daxpy( a[1], &bs[ 9][0], c );  a[1] = A[1*lda];
            daxpy( a[2], &bs[10][0], c );  a[2] = A[2*lda];
            daxpy( a[3], &bs[11][0], c );  a[3] = A[3*lda];

            A += 4*lda;
            daxpy( a[0], &bs[12][0], c );
            daxpy( a[1], &bs[13][0], c );
            daxpy( a[2], &bs[14][0], c );
            daxpy( a[3], &bs[15][0], c );

            B += 16;
            __syncthreads();
        } while( B < Blast );

        for( int i = 0; i < 16; i++ ) {
            C[0] = c[i];
            C += ldc;
        }
    }
}

/*
 * B21 = -inv(A22)*A21*inv(A11)
 */
__global__ void
triple_zgemm_update_above64_part1_L (const magmaDoubleComplex *Ain, magmaDoubleComplex *d_dinvA, int blk, int lda, int npages)
{
    const int bIdy = blockIdx.y/npages;
    const int page = qmod(blockIdx.y, npages);
    const int inx = threadIdx.x;
    const int iny = threadIdx.y;
    const int ibx = blockIdx.x*64;
    const int iby = bIdy*16;
    const int id = inx + iny*16;
    __shared__ magmaDoubleComplex bs[16][17];

    int PagesPerNB = NB/(blk*2);
    //--------------------------part one---------------------------//
    {
        // A21*inv(A11) -> A21
        // A=A21, B=inv(A11), C=A21
        const magmaDoubleComplex *A;
        magmaDoubleComplex *B, *C;
        int ldb = NB;
        int ldc = NB;

        d_dinvA += NB*NB*(page/PagesPerNB)
                + (qmod(page, PagesPerNB))*(blk*2)*NB
                + (qmod(page, PagesPerNB))*(blk*2);

        A = Ain + page*lda*blk*2 + page*blk*2 + blk;
        B = d_dinvA;
        C = d_dinvA + blk;

        A += ibx + id;
        B += inx + __mul24( iby + iny, ldb );
        C += ibx + id  + __mul24( iby, ldc );

        const magmaDoubleComplex *Blast = B + blk;

        magmaDoubleComplex c[16] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

        do {
            magmaDoubleComplex a[4] = { A[0*lda], A[1*lda], A[2*lda], A[3*lda] };

            bs[inx][iny   ] = B[ 0*ldb];
            bs[inx][iny+ 4] = B[ 4*ldb];
            bs[inx][iny+ 8] = B[ 8*ldb];
            bs[inx][iny+12] = B[12*ldb];
            __syncthreads();

            A += 4*lda;
            daxpy( a[0], &bs[ 0][0], c );  a[0] = A[0*lda];
            daxpy( a[1], &bs[ 1][0], c );  a[1] = A[1*lda];
            daxpy( a[2], &bs[ 2][0], c );  a[2] = A[2*lda];
            daxpy( a[3], &bs[ 3][0], c );  a[3] = A[3*lda];

            A += 4*lda;
            daxpy( a[0], &bs[ 4][0], c );  a[0] = A[0*lda];
            daxpy( a[1], &bs[ 5][0], c );  a[1] = A[1*lda];
            daxpy( a[2], &bs[ 6][0], c );  a[2] = A[2*lda];
            daxpy( a[3], &bs[ 7][0], c );  a[3] = A[3*lda];

            A += 4*lda;
            daxpy( a[0], &bs[ 8][0], c );  a[0] = A[0*lda];
            daxpy( a[1], &bs[ 9][0], c );  a[1] = A[1*lda];
            daxpy( a[2], &bs[10][0], c );  a[2] = A[2*lda];
            daxpy( a[3], &bs[11][0], c );  a[3] = A[3*lda];

            A += 4*lda;
            daxpy( a[0], &bs[12][0], c );
            daxpy( a[1], &bs[13][0], c );
            daxpy( a[2], &bs[14][0], c );
            daxpy( a[3], &bs[15][0], c );

            B += 16;
            __syncthreads();
        } while( B < Blast );

        for( int i = 0; i < 16; i++ ) {
            C[0] = c[i];
            C += ldc;
        }
    }
}

/*
 * B21 = -inv(A11)*A12*inv(A22)
 */
__global__ void
triple_zgemm_update_above64_part2_R (const magmaDoubleComplex *Ain, magmaDoubleComplex *d_dinvA, int blk, int lda, int npages)
{
    const int bIdy = blockIdx.y/npages;
    const int page = qmod(blockIdx.y, npages);
    const int inx = threadIdx.x;
    const int iny = threadIdx.y;
    const int ibx = blockIdx.x*64;
    const int iby = bIdy*16;
    const int id = inx + iny*16;
    __shared__ magmaDoubleComplex bs[16][17];

    int PagesPerNB = NB/(blk*2);

    //--------------------------part two---------------------------//
    {
        // -inv(A11)*A12 -> A12
        // A=inv(A11), B=A12, C=A12
        const magmaDoubleComplex *A;
        magmaDoubleComplex *B, *C;
        int lda = NB;
        int ldb = NB;
        int ldc = NB;

        d_dinvA += NB*NB*(page/PagesPerNB)
                + (qmod(page, PagesPerNB))*(blk*2)*NB
                + (qmod(page, PagesPerNB))*(blk*2);

        A = d_dinvA;
        B = d_dinvA + blk*NB;
        C = d_dinvA + blk;

        A += ibx + id;
        B += inx + __mul24( iby + iny, ldb );
        C += ibx + id  + __mul24( iby, ldc );

        const magmaDoubleComplex *Blast = B + blk;

        magmaDoubleComplex c[16] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

        do {
            magmaDoubleComplex a[4] = { A[0*lda], A[1*lda], A[2*lda], A[3*lda] };

            bs[inx][iny   ] = B[ 0*ldb];
            bs[inx][iny+ 4] = B[ 4*ldb];
            bs[inx][iny+ 8] = B[ 8*ldb];
            bs[inx][iny+12] = B[12*ldb];
            __syncthreads();

            A += 4*lda;
            daxpy( a[0], &bs[ 0][0], c );  a[0] = A[0*lda];
            daxpy( a[1], &bs[ 1][0], c );  a[1] = A[1*lda];
            daxpy( a[2], &bs[ 2][0], c );  a[2] = A[2*lda];
            daxpy( a[3], &bs[ 3][0], c );  a[3] = A[3*lda];

            A += 4*lda;
            daxpy( a[0], &bs[ 4][0], c );  a[0] = A[0*lda];
            daxpy( a[1], &bs[ 5][0], c );  a[1] = A[1*lda];
            daxpy( a[2], &bs[ 6][0], c );  a[2] = A[2*lda];
            daxpy( a[3], &bs[ 7][0], c );  a[3] = A[3*lda];

            A += 4*lda;
            daxpy( a[0], &bs[ 8][0], c );  a[0] = A[0*lda];
            daxpy( a[1], &bs[ 9][0], c );  a[1] = A[1*lda];
            daxpy( a[2], &bs[10][0], c );  a[2] = A[2*lda];
            daxpy( a[3], &bs[11][0], c );  a[3] = A[3*lda];

            A += 4*lda;
            daxpy( a[0], &bs[12][0], c );
            daxpy( a[1], &bs[13][0], c );
            daxpy( a[2], &bs[14][0], c );
            daxpy( a[3], &bs[15][0], c );

            B += 16;
            __syncthreads();
        } while( B < Blast );

        for( int i = 0; i < 16; i++ ) {
            C[0] = (-1)*c[i];
            C += ldc;
        }
    }
}

/*
 * part 3, copy data into position
 */
__global__ void
triple_zgemm_update_above64_part3_R (const magmaDoubleComplex *Ain, magmaDoubleComplex *d_dinvA, int blk, int lda, int npages)
{
    const int bIdy = blockIdx.y/npages;
    const int page = qmod(blockIdx.y, npages);
    const int inx = threadIdx.x;
    const int iny = threadIdx.y;
    const int ibx = blockIdx.x*64;
    const int iby = bIdy*16;
    const int id = inx + iny*16;

    int PagesPerNB = NB/(blk*2);

    //--------------------------part two---------------------------//
    {
        // -inv(A11)*A12 -> A12
        // A=inv(A11), B=A12, C=A12
        magmaDoubleComplex *C_temp, *C_real;
        int ldc = NB;

        C_temp = d_dinvA + NB*NB*(page/PagesPerNB)
               + (qmod(page, PagesPerNB))*(blk*2)*NB
               + (qmod(page, PagesPerNB))*(blk*2)
               + blk;

        C_real = d_dinvA + NB*NB*(page/PagesPerNB)
               + (qmod(page, PagesPerNB))*(blk*2)*NB
               + blk*NB
               + (qmod(page, PagesPerNB))*(blk*2);

        C_temp += ibx + id  + __mul24( iby, ldc );
        C_real += ibx + id  + __mul24( iby, ldc );

        for( int i = 0; i < 16; i++ ) {
            C_real[0] = C_temp[0];
            C_temp[0] = MAGMA_Z_ZERO;
            C_temp += ldc;
            C_real += ldc;
        }
    }
}

/*
 * part 3: copy data back to position
 */
__global__ void
triple_zgemm_update_above64_part3_L (const magmaDoubleComplex *Ain, magmaDoubleComplex *d_dinvA, int blk, int lda, int npages)
{
    const int bIdy = blockIdx.y/npages;
    const int page = qmod(blockIdx.y, npages);
    const int inx = threadIdx.x;
    const int iny = threadIdx.y;
    const int ibx = blockIdx.x*64;
    const int iby = bIdy*16;
    const int id = inx + iny*16;

    int PagesPerNB = NB/(blk*2);

    //--------------------------part three---------------------------//
    {
        // -inv(A22)*A21 -> A21
        // A=inv(A22), B=A21, C=A21
        magmaDoubleComplex *C_temp, *C_real;
        int ldc = NB;

        d_dinvA += NB*NB*(page/PagesPerNB)
                + (qmod(page, PagesPerNB))*(blk*2)*NB
                + (qmod(page, PagesPerNB))*(blk*2);

        C_real = d_dinvA + blk;

        C_temp = d_dinvA + blk*NB;

        C_temp += ibx + id  + __mul24( iby, ldc );
        C_real += ibx + id  + __mul24( iby, ldc );

        for( int i = 0; i < 16; i++ ) {
            C_real[0] = C_temp[0];
            C_temp[0] = MAGMA_Z_ZERO;
            C_real += ldc;
            C_temp += ldc;
        }
    }
    __syncthreads();
}

/*
 * B21 = -inv(A22)*A21*inv(A11)
 */
__global__ void
triple_zgemm_update_above64_part2_L (const magmaDoubleComplex *Ain, magmaDoubleComplex *d_dinvA, int blk, int lda, int npages)
{
    const int bIdy = blockIdx.y/npages;
    const int page = qmod(blockIdx.y, npages);
    const int inx = threadIdx.x;
    const int iny = threadIdx.y;
    const int ibx = blockIdx.x*64;
    const int iby = bIdy*16;
    const int id = inx + iny*16;
    __shared__ magmaDoubleComplex bs[16][17];

    int PagesPerNB = NB/(blk*2);

    //--------------------------part two---------------------------//
    {
        // -inv(A22)*A21 -> A21
        // A=inv(A22), B=A21, C=A21
        magmaDoubleComplex *A, *B, *C;
        int lda = NB;
        int ldb = NB;
        int ldc = NB;

        d_dinvA += NB*NB*(page/PagesPerNB)
                + (qmod(page, PagesPerNB))*(blk*2)*NB
                + (qmod(page, PagesPerNB))*(blk*2);

        A = d_dinvA + blk*NB + blk;
        B = d_dinvA + blk;

        C = d_dinvA + blk*NB;

        A += ibx + id;
        B += inx + __mul24( iby + iny, ldb );
        C += ibx + id  + __mul24( iby, ldc );

        const magmaDoubleComplex *Blast = B + blk;

        magmaDoubleComplex c[16] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

        do {
            magmaDoubleComplex a[4] = { A[0*lda], A[1*lda], A[2*lda], A[3*lda] };

            bs[inx][iny   ] = B[ 0*ldb];
            bs[inx][iny+ 4] = B[ 4*ldb];
            bs[inx][iny+ 8] = B[ 8*ldb];
            bs[inx][iny+12] = B[12*ldb];
            __syncthreads();

            A += 4*lda;
            daxpy( a[0], &bs[ 0][0], c );  a[0] = A[0*lda];
            daxpy( a[1], &bs[ 1][0], c );  a[1] = A[1*lda];
            daxpy( a[2], &bs[ 2][0], c );  a[2] = A[2*lda];
            daxpy( a[3], &bs[ 3][0], c );  a[3] = A[3*lda];

            A += 4*lda;
            daxpy( a[0], &bs[ 4][0], c );  a[0] = A[0*lda];
            daxpy( a[1], &bs[ 5][0], c );  a[1] = A[1*lda];
            daxpy( a[2], &bs[ 6][0], c );  a[2] = A[2*lda];
            daxpy( a[3], &bs[ 7][0], c );  a[3] = A[3*lda];

            A += 4*lda;
            daxpy( a[0], &bs[ 8][0], c );  a[0] = A[0*lda];
            daxpy( a[1], &bs[ 9][0], c );  a[1] = A[1*lda];
            daxpy( a[2], &bs[10][0], c );  a[2] = A[2*lda];
            daxpy( a[3], &bs[11][0], c );  a[3] = A[3*lda];

            A += 4*lda;
            daxpy( a[0], &bs[12][0], c );
            daxpy( a[1], &bs[13][0], c );
            daxpy( a[2], &bs[14][0], c );
            daxpy( a[3], &bs[15][0], c );

            B += 16;
            __syncthreads();
        } while( B < Blast );

        for( int i = 0; i < 16; i++ ) {
            C[0] = (-1)*c[i];
            C += ldc;
        }
    }
}


extern "C"
void diag_ztrtri (magma_int_t M, 
magma_uplo_t uplo, 
magma_diag_t diag, 
const magmaDoubleComplex *A, 
magmaDoubleComplex *d_dinvA, 
magma_int_t lda)
{
/*
   This routine is used in ztrsm
*/

    int nblocks = M/BLOCK_SIZE + (M % BLOCK_SIZE != 0);

    if (uplo == MagmaLower) {
        // solve the diagonal blocks
        diag_ztrtri_kernel_lower<<< nblocks, BLOCK_SIZE, 0, magma_stream >>>(diag, A, d_dinvA, lda);

        // update the inverse up to the size of BLOCK_SIZE
        for( int i=BLOCK_SIZE; i < NB; i*=2 ) {
            int npages = M/(i*2)+(M%(i*2)!=0);
            dim3 dimBlock((i <= 32)?(i/4):16, 4);
            dim3 dimGrid(i/(dimBlock.x*dimBlock.y), npages*(i/16));    // emulated 3D grid, see 3d_grid.txt

            switch (i) {
                case 16:
                    triple_zgemm_update_16_part1_L<<< dimGrid, dimBlock, 0, magma_stream >>>(A, d_dinvA, i, lda, npages);
                    triple_zgemm_update_16_part2_L<<< dimGrid, dimBlock, 0, magma_stream >>>(A, d_dinvA, i, lda, npages);
                    break;
                case 32:
                    triple_zgemm_update_32_part1_L<<< dimGrid, dimBlock, 0, magma_stream >>>(A, d_dinvA, i, lda, npages);
                    triple_zgemm_update_32_part2_L<<< dimGrid, dimBlock, 0, magma_stream >>>(A, d_dinvA, i, lda, npages);
                    break;
                case 64:
                    triple_zgemm_update_64_part1_L<<< dimGrid, dimBlock, 0, magma_stream >>>(A, d_dinvA, i, lda, npages);
                    triple_zgemm_update_64_part2_L<<< dimGrid, dimBlock, 0, magma_stream >>>(A, d_dinvA, i, lda, npages);
                    break;
                default:
                    triple_zgemm_update_above64_part1_L<<< dimGrid, dimBlock, 0, magma_stream >>>(A, d_dinvA, i, lda, npages);
                    triple_zgemm_update_above64_part2_L<<< dimGrid, dimBlock, 0, magma_stream >>>(A, d_dinvA, i, lda, npages);
                    triple_zgemm_update_above64_part3_L<<< dimGrid, dimBlock, 0, magma_stream >>>(A, d_dinvA, i, lda, npages);
                    break;
            }
            if (i*2 >= M) break;
        }
    }
    else {
        diag_ztrtri_kernel_upper<<< nblocks, BLOCK_SIZE, 0, magma_stream >>>(diag, A, d_dinvA, lda);

        // update the inverse up to the size of BLOCK_SIZE
        for( int i=BLOCK_SIZE; i < NB; i*=2 ) {
            int npages = M/(i*2)+(M%(i*2)!=0);
            dim3 dimBlock((i <= 32)?(i/4):16, 4);
            dim3 dimGrid(i/(dimBlock.x*dimBlock.y), npages*(i/16));    // emulated 3D grid, see 3d_grid.txt

            switch (i) {
                case 16:
                    triple_zgemm_update_16_R<<< dimGrid, dimBlock, 0, magma_stream >>>(A, d_dinvA, i, lda, npages);
                    break;
                case 32:
                    triple_zgemm_update_32_part1_R<<< dimGrid, dimBlock, 0, magma_stream >>>(A, d_dinvA, i, lda, npages);
                    triple_zgemm_update_32_part2_R<<< dimGrid, dimBlock, 0, magma_stream >>>(A, d_dinvA, i, lda, npages);
                    break;
                case 64:
                    triple_zgemm_update_64_part1_R<<< dimGrid, dimBlock, 0, magma_stream >>>(A, d_dinvA, i, lda, npages);
                    triple_zgemm_update_64_part2_R<<< dimGrid, dimBlock, 0, magma_stream >>>(A, d_dinvA, i, lda, npages);
                    break;
                default:
                    triple_zgemm_update_above64_part1_R<<< dimGrid, dimBlock, 0, magma_stream >>>(A, d_dinvA, i, lda, npages);
                    triple_zgemm_update_above64_part2_R<<< dimGrid, dimBlock, 0, magma_stream >>>(A, d_dinvA, i, lda, npages);
                    triple_zgemm_update_above64_part3_R<<< dimGrid, dimBlock, 0, magma_stream >>>(A, d_dinvA, i, lda, npages);
                    break;
            }
            if (i*2 >= M) break;
        }
    }
}

