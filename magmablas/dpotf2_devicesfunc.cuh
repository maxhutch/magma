
/*
   -- MAGMA (version 2.0.2) --
   Univ. of Tennessee, Knoxville
   Univ. of California, Berkeley
   Univ. of Colorado, Denver
   @date May 2016

   @author Azzam Haidar
   @author Ahmad Ahmad

   @generated from magmablas/zpotf2_devicesfunc.cuh normal z -> d, Mon May  2 23:31:25 2016
 */


#ifndef MAGMABLAS_DPOTF2_DEVICES_Z_H
#define MAGMABLAS_DPOTF2_DEVICES_Z_H


extern __shared__ double shared_data[];
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
static inline __device__ void dpotf2_sminout_anywidth_device(const int m, const int n, double *A, const int lda)
{
    const int tx = threadIdx.x;
    double factor;
    #pragma unroll
    for (int iter=0; iter < n; iter++)
    {
        //sqrt(diag) and dscal
        #ifdef ENABLE_COND1
        if ( tx >= iter && tx < m )
        {
        #endif
            double xreal = sqrt(MAGMA_D_REAL(A[iter + iter * lda]));
            factor = MAGMA_D_MAKE(1.0/xreal, 0.0);
        #ifdef ENABLE_COND1
        }
        #endif
        __syncthreads();
        #ifdef ENABLE_COND1
        if ( tx >= iter && tx < m )
        {
        #endif
            A[ tx + iter * lda ] *= factor;
            //A[ tx + iter * lda ]  = tx == iter ? MAGMA_D_MAKE(xreal, 0.0) : A[ tx + iter * lda ] * factor;
        #ifdef ENABLE_COND1
        }
        #endif
        __syncthreads();


        // dlacgv: TODO, dsyrk
        #ifdef ENABLE_COND1
        if ( tx > iter && tx < m )
        {
        #endif
            #pragma unroll 
            for (int j=iter+1; j < n; j++)
            {
                A [tx + j * lda] -= A[tx + iter * lda]  *  MAGMA_D_CONJ(A[iter * lda + j]);
            }   
        #ifdef ENABLE_COND1
        }
        #endif
        __syncthreads();
    }// end of iter
}
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
static inline __device__ void dpotf2_sminout_fixsize_device(const int m, double *A, const int lda)
{
    const int tx = threadIdx.x;
    double factor;
    //__shared__ double row[POTF2_NB];

    #pragma unroll
    for (int iter=0; iter < POTF2_NB; iter++)
    {
        //sqrt(diag) and dscal
        #ifdef ENABLE_COND2
        if ( tx >= iter && tx < m )
        {
        #endif
            double xreal = sqrt(MAGMA_D_REAL(A[iter + iter * lda]));
            factor = MAGMA_D_MAKE(1.0/xreal, 0.0);
        #ifdef ENABLE_COND2
        }
        #endif
        __syncthreads();
        #ifdef ENABLE_COND2
        if ( tx >= iter && tx < m )
        {
        #endif
            A[ tx + iter * lda ] *= factor;

            //A[ tx + iter * lda ]  = tx == iter ? MAGMA_D_MAKE(xreal, 0.0) : A[ tx + iter * lda ] * factor;
            //if (tx < POTF2_NB) row[ tx ] = MAGMA_D_CONJ( A[ tx + iter * lda ] );
            //if (tx < POTF2_NB) A[ iter + tx * lda ] = MAGMA_D_CONJ( A[ tx + iter * lda ] );
        #ifdef ENABLE_COND2
        }
        #endif

        __syncthreads();


        // dsyrk
        #ifdef ENABLE_COND2
        if ( tx > iter && tx < m )
        {
        #endif
            #pragma unroll
            for (int j=iter+1; j < POTF2_NB; j++)
            {
                A [tx + j * lda] -= A[tx + iter * lda]  *  MAGMA_D_CONJ(A[iter * lda + j]);
                //A [tx + j * lda] -= A[tx + iter * lda]  *  row[j];
                //A [tx + j * lda] -= A[tx + iter * lda]  *  A[iter +lda * j];
            }   
        #ifdef ENABLE_COND2
        }
        #endif
        __syncthreads();
    }// end of iter
}
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


/////////////////////////////////////////////////////////////////////////////////////////////////
static inline __device__ void dgemm_v20_1_fixsize_device(int m, int k,
        const double* __restrict__ A0, const int lda,
        double *sC, double  *sB)
{
    const int tx = threadIdx.x;
    double rC[POTF2_NB];
    double rA[POTF2_NB]; 
    double rp[POTF2_NB]; 

    // prefetch next block. 
    #ifdef ENABLE_COND4
    if (tx < m) 
    {
    #endif
        #pragma unroll
        for (int i=0; i < POTF2_NB; i++)
        {
            rp[i] = A0[tx + i * lda];
        }
    #ifdef ENABLE_COND4
    }
    #endif

    #pragma unroll
    for (int i=0; i < POTF2_NB; i++)
    {
        rC[i] = MAGMA_D_ZERO;
    }
    __syncthreads();



    // accumulate 
    #pragma unroll
    for (int iter=0; iter < k; iter += POTF2_NB)
    {
        #ifdef ENABLE_COND4
        if (tx < m) 
        {
        #endif
            // rp to rA
            #pragma unroll
            for (int i=0; i < POTF2_NB; i++)
            {
                rA[i] = rp[i];
            }
        #ifdef ENABLE_COND4
        }
        #endif

        // rA to sB
        if (tx < POTF2_NB) 
        {      
            #pragma unroll
            for (int i=0; i < POTF2_NB; i++)
            {
                sB[tx + i * POTF2_NB] = MAGMA_D_CONJ(rp[i]);
            }
        }

        __syncthreads();

        // prefetch next block. Azzam
        #ifdef ENABLE_COND4
        if (tx < m )  
        {      
        #endif
            #pragma unroll
            for (int i=0; i < POTF2_NB; i++)
            {
                rp[i] = A0[tx + (i+(iter+POTF2_NB)) * lda];
            }
        #ifdef ENABLE_COND4
        }
        #endif
        //__syncthreads();

        // multiply current block
        #ifdef ENABLE_COND4
        if (tx < m) 
        {
        #endif
            for (int i=0; i < POTF2_NB; i++)
            {
                #pragma unroll
                for (int col=0; col < POTF2_NB; col++)
                {
                    // A0 is multiplied by POTF2_NB times
                    rC[col] +=  rA[i] * sB[col + i * POTF2_NB];
                }
            }
        #ifdef ENABLE_COND4
        }
        #endif
        __syncthreads();
    }//end of accumulation

    // finalyzing gemm.
    #ifdef ENABLE_COND4
    if (tx < m) 
    {
    #endif
        #pragma unroll
        for (int i=0; i < POTF2_NB; i++)
        {
            sC[tx + i *m] = rp[i] - rC[i];
        }
    #ifdef ENABLE_COND4
    }
    #endif
    __syncthreads();
}
/////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////
static inline __device__ void dgemm_v20_1_anywidth_device(int m, int n, int k,
        const double* __restrict__ A0, int lda,
        double *sC, double  *sB)
{
    const int tx = threadIdx.x;
    double rC[POTF2_NB];
    double rA[POTF2_NB]; 
    double rp[POTF2_NB]; 

    const int bound_A = lda*(k+n-1)+m;

    // prefetch next block. 
    #ifdef ENABLE_COND5
    if (tx < m) 
    {
    #endif
        #pragma unroll
        for (int i=0; i < POTF2_NB; i++)
        {
            rp[i] = A0[min(bound_A, tx + i * lda)];
        }
    #ifdef ENABLE_COND5
    }
    #endif

    #pragma unroll
    for (int i=0; i < POTF2_NB; i++)
    {
        rC[i] = MAGMA_D_ZERO;
    }
    __syncthreads();



    // accumulate 
    #pragma unroll
    for (int iter=0; iter < k; iter += POTF2_NB)
    {
        #ifdef ENABLE_COND5
        if (tx < m) 
        {
        #endif
            // rp to rA
            #pragma unroll
            for (int i=0; i < POTF2_NB; i++)
            {
                rA[i] = rp[i];
            }
        #ifdef ENABLE_COND5
        }
        #endif

        // rA to sB
        if (tx < POTF2_NB) 
        {      
            #pragma unroll
            for (int i=0; i < POTF2_NB; i++)
            {
                sB[tx + i * POTF2_NB] = MAGMA_D_CONJ(rp[i]);
            }
        }

        __syncthreads();

        // prefetch next block. Azzam
        #ifdef ENABLE_COND5
        if (tx < m )  
        {      
        #endif
            #pragma unroll
            for (int i=0; i < POTF2_NB; i++)
            {
                rp[i] = A0[min(bound_A, tx + (i+(iter+POTF2_NB)) * lda)]; // min(bound,xxx) is to avoid reading out of bound
            }
        #ifdef ENABLE_COND5
        }
        #endif
        //__syncthreads();

        // multiply current block
        #ifdef ENABLE_COND5
        if (tx < m) 
        {
        #endif
            for (int i=0; i < POTF2_NB; i++)
            {
                #pragma unroll
                for (int col=0; col < POTF2_NB; col++)
                {
                    // A0 is multiplied by POTF2_NB times
                    rC[col] +=  rA[i] * sB[col + i * POTF2_NB];
                }
            }
        #ifdef ENABLE_COND5
        }
        #endif
        __syncthreads();
    }//end of accumulation

    // finalyzing gemm.
    #ifdef ENABLE_COND5
    if (tx < m) 
    {
    #endif
        #pragma unroll
        for (int i=0; i < POTF2_NB; i++)
        {
            sC[tx + i *m] = rp[i] - rC[i];
        }
    #ifdef ENABLE_COND5
    }
    #endif
    __syncthreads();
}
/////////////////////////////////////////////////////////////////////////////////////////////////





/////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////
static inline __device__ void dpotf2_smlpout_fixwidth_device(const int m,  
        double *A0, double *A, int lda,
        const int localstep, const int gbstep,
        magma_int_t *info)
{
    // checkinfo to avoid computation of the singular matrix
    #ifndef BATCH_DISABLE_CHECKING
    if (*info != 0 ) return;
    #endif

    const int tx = threadIdx.x;
    double *sdata_A = shared_data + threadIdx.y * (m+POTF2_NB)*POTF2_NB;
    double *sdata_B = sdata_A + m * POTF2_NB;


    #if 1
    dgemm_v20_1_fixsize_device(m, localstep, 
                       A0, lda, sdata_A, sdata_B);
    #else
    dgemm_v20_1_anywidth_device(m, POTF2_NB, localstep, 
                       A0, lda, sdata_A, sdata_B);
    #endif
    //----------------------------------------------------
    // Check for not SPD generating info
    #ifndef BATCH_DISABLE_CHECKING
    const int ty = threadIdx.y;
    __shared__ int cleanup[MAX_NTCOL];
    if ( tx == 0) {
        cleanup[ty] = 0;
        #pragma unroll
        for (int i=0; i < POTF2_NB; i++)
        {
            if (MAGMA_D_REAL(sdata_A[i + i * m]) <= MAGMA_D_ZERO )
            {
                #if 0
                if (cleanup[ty] == 0) *info = i + gbstep + 1;
                cleanup[ty] = 1;
                #else
                *info = i + gbstep + 1;
                cleanup[ty] = 1;
                break;
                #endif
            }
        }
    }
    __syncthreads();
    if (cleanup[ty] == 1) return;
    #endif
    //----------------------------------------------------

    dpotf2_sminout_fixsize_device(m, sdata_A, m);
    //dpotf2_sminout_anywidth_device(m, POTF2_NB, sdata_A, m);

    //copy sdata_A to A
    #ifdef ENABLE_COND6
    if (tx < m)
    {
    #endif
        #pragma unroll
        for (int i=0; i < POTF2_NB; i++)
        {  
            #ifdef BATCH_DISABLE_CLEANUP
            A[tx + i * lda] = sdata_A[tx + i * m];
            #else
            if (tx >= i) A[tx + i * lda] = sdata_A[tx + i * m];
            #endif
        }
    #ifdef ENABLE_COND6
    }
    #endif
}
/////////////////////////////////////////////////////////////////////////////////////////////////
static inline __device__ void dpotf2_smlpout_anywidth_device(const int m, const int n,
        double *A0, double *A, int lda,
        const int localstep, const int gbstep,
        magma_int_t *info)
{
    // checkinfo to avoid computation of the singular matrix
    #ifndef BATCH_DISABLE_CHECKING
    if (*info != 0 ) return;
    #endif
    const int tx = threadIdx.x;
    double *sdata_A = shared_data + threadIdx.y * (m+POTF2_NB)*POTF2_NB;
    double *sdata_B = sdata_A + m * POTF2_NB;

    #if 0
    dgemm_v20_1_fixsize_device(m, localstep, 
                       A0, lda, sdata_A, sdata_B);
    dpotf2_sminout_fixsize_device(m, sdata_A, m);
    #else
    dgemm_v20_1_anywidth_device(m, n, localstep, 
                       A0, lda, sdata_A, sdata_B);
    //----------------------------------------------------
    // Check for not SPD generating info
    #ifndef BATCH_DISABLE_CHECKING
    const int ty = threadIdx.y;
    __shared__ int cleanup[MAX_NTCOL];
    if ( tx == 0) {
        cleanup[ty] = 0;
        #pragma unroll
        for (int i=0; i < n; i++)
        {
            if (MAGMA_D_REAL(sdata_A[i + i * m]) <= MAGMA_D_ZERO )
            {
                #if 0
                if (cleanup[ty] == 0) *info = i + gbstep + 1;
                cleanup[ty] = 1;
                #else
                *info = i + gbstep + 1;
                cleanup[ty] = 1;
                break;
                #endif
            }
        }
    }
    __syncthreads();
    if (cleanup[ty] == 1) return;
    #endif
    //----------------------------------------------------
    dpotf2_sminout_anywidth_device(m, n, sdata_A, m);
    #endif


    //copy sdata_A to A
    #ifdef ENABLE_COND6
    if (tx < m)
    {
    #endif
        #pragma unroll
        for (int i=0; i < n; i++)
        {  
            #ifdef BATCH_DISABLE_CLEANUP
            A[tx + i * lda] = sdata_A[tx + i * m];
            #else
            if (tx >= i) A[tx + i * lda] = sdata_A[tx + i * m];
            #endif
        }
    #ifdef ENABLE_COND6
    }
    #endif        
}
/////////////////////////////////////////////////////////////////////////////////////////////////



#endif  /* MAGMABLAS_DPOTF2_DEVICES_Z_H */
