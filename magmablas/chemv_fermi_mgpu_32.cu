/*
    -- MAGMA (version 1.4.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       August 2013

       @generated c Wed Aug 14 12:16:43 2013

*/
#include "common_magma.h"
#define PRECISION_c

#if (GPUSHMEM >= 200) || defined(PRECISION_s) || defined(PRECISION_d) || defined(PRECISION_c)

#define chemv_bs         32
#define bank_shift       33

/*******************************************************************************
 *     Functions for each specific cases - Lower case
 */

__global__ void
magmablas_chemv_200_L_special_mgpu_offset_32(
    int n, magmaFloatComplex alpha,
    magmaFloatComplex *A, int lda,
    magmaFloatComplex *x, int incx,
    magmaFloatComplex  beta,
    magmaFloatComplex *y, int incy,
    magmaFloatComplex *WC,
    int my_gpu_id,
    int num_gpus,
    int nb,
    int kstan)
{
    int tx   = threadIdx.x;
    int ty   = threadIdx.y;
    int blkc = blockIdx.x;

    if(blkc < my_gpu_id)
    {
        return;
    }

    magmaFloatComplex res  = MAGMA_C_ZERO; // used in scan the row
    magmaFloatComplex res_ = MAGMA_C_ZERO; // used in scan the column
    magmaFloatComplex res1 = MAGMA_C_ZERO; // tem for res
    magmaFloatComplex res2 = MAGMA_C_ZERO; // tem for res_

    __shared__ magmaFloatComplex la   [chemv_bs][bank_shift];
    __shared__ magmaFloatComplex sdata   [chemv_bs][9];
    __shared__ magmaFloatComplex buff [chemv_bs];
    __shared__ magmaFloatComplex buff2 [chemv_bs];

    int break_d   =  chemv_bs * blkc;

    x  += (break_d + tx ) * incx;
    A  +=  break_d;
    A  +=  ty * lda + tx;

    if( ty == 0 )
    {
        buff[tx] = x[0];
        if(blkc == 0 && my_gpu_id == 0 && tx < kstan)
        {
            MAGMA_C_SET2REAL(buff[tx], 0.0);
        }
    } // obtain the vector x store in buff;

    int flag = 0;
    
    if ( (blkc % num_gpus) == my_gpu_id)
    {
        A += lda * (blkc/num_gpus) * chemv_bs; // change

        #pragma unroll
        for(int j =0; j < chemv_bs; j += 8)
            la[0][ bank_shift * (ty+j) + tx] =  A[ j * lda];
        __syncthreads();

        #pragma unroll
        for(int  i=ty*4; i < (ty * 4 + 4); i++) {
            if ( i < tx ) {
                la[0][bank_shift * tx + i] = cuConjf( la[0][ i * bank_shift + tx] );
            }
        }
        __syncthreads();

        #pragma unroll
        for(int j=0; j < 4; j++)
            res += cuConjf( la[0][bank_shift * tx + j + ty * 4] ) * buff[j + ty * 4];
        __syncthreads();

        A -= lda * (blkc/num_gpus) * chemv_bs;
            
        flag = 1;
    }

    x -= blkc * chemv_bs  *incx;

    x= x- tx*incx;

    int wc_c = my_gpu_id;
    int count = 0;

    WC +=  break_d + tx;
    
    int num_blocks_iters = (blkc +1) /num_gpus - flag;
    
    if( my_gpu_id < ( (blkc+1) % num_gpus) )
    {
        num_blocks_iters += 1;
    }

    x += (my_gpu_id ) * chemv_bs;

    if( blkc > my_gpu_id) {
        for(int s=0; s < num_blocks_iters; s++)
        {
            MAGMA_C_SET2REAL(res_,0);
            count++;

            #pragma unroll
            for(int j =0; j < chemv_bs; j += 8)
                la[0][ bank_shift * (ty+j) + tx] =  A[ j * lda];

            if( ty == 0 )
            {
                buff2[tx] = x[tx];
                if(my_gpu_id == 0 && tx < kstan && count == 1)
                {
                    MAGMA_C_SET2REAL(buff2[tx], 0.0);
                }
            } // obtain the vector x store in buff2;
            __syncthreads();

            #pragma unroll
            for(int j=0; j < 4; j++)
            {
                res += (la[0][bank_shift * (ty + j * 8) + tx] )* buff2[ ty + j * 8];
                res_ += cuConjf( la[0][bank_shift * tx + j + ty * 4] ) * buff[j + ty * 4]; //iterate colum
            }

            sdata[tx][ty]= res_;
            __syncthreads();

            if( ty == 1 )
            {
                res2 = sdata[tx][0]+sdata[tx][1]
                     + sdata[tx][2]+sdata[tx][3]
                     + sdata[tx][4]+sdata[tx][5]
                     + sdata[tx][6]+sdata[tx][7];
                
                WC[wc_c*lda ] =   res2;
            }

            wc_c += num_gpus;
            x += num_gpus * chemv_bs;
            A += lda * chemv_bs;
        }
    }

    la[0][bank_shift*tx+ty]= res;
    __syncthreads();
    
    if( ty == 0 )
    {
        res1 = la[0][tx*bank_shift+0]+la[0][tx*bank_shift+1]
             + la[0][tx*bank_shift+2]+la[0][tx*bank_shift+3]
             + la[0][tx*bank_shift+4]+la[0][tx*bank_shift+5]
             + la[0][tx*bank_shift+6]+la[0][tx*bank_shift+7];
        
        WC[0+lda*(blkc)] =  res1;
    }
}

/**************************************************************
 *    Lower case for generic sizes
 */
__global__ void
magmablas_chemv_200_L_generic_mgpu_offset_32(
    int n, magmaFloatComplex alpha,
    magmaFloatComplex *A, int lda,
    magmaFloatComplex *x, int incx,
    magmaFloatComplex beta,
    magmaFloatComplex *y, int incy,
    magmaFloatComplex *WC,
    int m_mod_nb,
    int my_gpu_id,
    int num_gpus,
    int nb,
    int kstan)
{
    int tx   = threadIdx.x;
    int ty   = threadIdx.y;
    int blkc = blockIdx.x;

    if(blkc < my_gpu_id)
    {
        return;
    }

    magmaFloatComplex res  = MAGMA_C_ZERO;
    magmaFloatComplex res_ = MAGMA_C_ZERO;
    magmaFloatComplex res1 = MAGMA_C_ZERO;
    magmaFloatComplex res2 = MAGMA_C_ZERO;

    __shared__ magmaFloatComplex la   [chemv_bs][bank_shift];
    __shared__ magmaFloatComplex sdata   [chemv_bs][9];
    __shared__ magmaFloatComplex buff [chemv_bs];
    __shared__ magmaFloatComplex buff2 [chemv_bs];

    int break_d   =  chemv_bs * blkc;

    x += (break_d + tx ) * incx;
    A +=  break_d;
    A += lda * ty;

    int trackA;
    if( blkc == ( gridDim.x - 1 ) ) {
        if( ty == 0 ) {
            if( tx > m_mod_nb )
            {
                MAGMA_C_SET2REAL(buff[tx],0);
            }
            else
                buff[tx]  = x[0];
        }
        if ( tx > m_mod_nb )
            trackA=m_mod_nb;
        else
            trackA=tx;
        A += trackA;
    }
    else {
        if( ty == 0 ) {
            buff[tx]  = x[0];
        }
        trackA = tx;
        A += trackA;
    }

    if(ty == 0 )
    {
        if(my_gpu_id == 0 && blkc == 0  && tx < kstan)//
        {
            MAGMA_C_SET2REAL(buff[tx], 0.0);
        }
    }

    int flag = 0;
    
    if ( (blkc % num_gpus) == my_gpu_id)
    {
        A += lda * (blkc/num_gpus) * chemv_bs; // change
        // Somehow merging these two if - else creates problem
        // It could be a potential bug -- from synchronization or from cuda or compiler
        if( blkc == ( gridDim.x - 1 ) ) {
            #pragma unroll
            for(int j =0; j < chemv_bs; j += 8) {
                if( ( ty + j ) > m_mod_nb )
                {
                    MAGMA_C_SET2REAL(la[0][bank_shift*(ty+j)+tx], 9999);
                }
                else
                    la[0][bank_shift*(ty+j)+tx] =  A[ j * lda];
            }
        }
        else {
            #pragma unroll
            for(int j =0; j < chemv_bs; j += 8) {
                la[0][bank_shift*(ty+j)+tx] = A[ j * lda];
            }
        }
        __syncthreads();

        #pragma unroll
        for(int  i=ty*4; i < (ty*4+4); i++) {
            if ( i < tx ) {
                la[0][bank_shift*tx+i] = cuConjf(la[0][i*bank_shift+tx]);
            }
        }
        __syncthreads();

        #pragma unroll
        for(int j=0; j < 4; j++)
            res += cuConjf(la[0][bank_shift*tx+j+ty*4])* buff[j+ty*4];
        __syncthreads();

        A -= lda * (blkc/num_gpus) * chemv_bs;
        
        flag = 1;
    }

    __syncthreads();

    x= x - break_d *incx;
    x= x - tx * incx;

    int wc_c = my_gpu_id;
    int count = 0;

    WC +=  break_d + tx;

    int num_blocks_iters = (blkc +1) /num_gpus - flag;
    
    if( my_gpu_id < ( (blkc+1) % num_gpus) )
    {
        num_blocks_iters += 1;
    }

    x += (my_gpu_id ) * chemv_bs;

    if( blkc > my_gpu_id) {
        for(int s=0; s < num_blocks_iters; s++)
        {
            MAGMA_C_SET2REAL(res_,0);
            count++;

            #pragma unroll
            for(int j =0; j < chemv_bs; j += 8)
                la[0][ bank_shift * (ty+j) + tx] =  A[ j * lda];

            if( ty == 0 )
            {
                buff2[tx] = x[tx];
                if(my_gpu_id == 0 && tx < kstan && count == 1)//
                {
                    MAGMA_C_SET2REAL(buff2[tx], 0.0);
                }
            } // obtain the vector x store in buff2;
            __syncthreads();

            #pragma unroll
            for(int j=0; j < 4; j++)
            {
                res += (la[0][bank_shift * (ty + j * 8) + tx] )* buff2[ ty + j * 8];
                res_ += cuConjf( la[0][bank_shift * tx + j + ty * 4] ) * buff[j + ty * 4]; //iterate colum
            }

            sdata[tx][ty]= res_;
            __syncthreads();

            if( ty == 1 )
            {
                res2 = sdata[tx][0]+sdata[tx][1]
                     + sdata[tx][2]+sdata[tx][3]
                     + sdata[tx][4]+sdata[tx][5]
                     + sdata[tx][6]+sdata[tx][7];
                
                WC[wc_c*lda ] =   res2;
            }

            wc_c += num_gpus;
            x += num_gpus * chemv_bs;
            A += lda * chemv_bs;
        }
    }

    la[0][bank_shift*tx+ty]= res;
    __syncthreads();
    
    if( ty == 0 )
    {
        res1 = la[0][tx*bank_shift+0]+la[0][tx*bank_shift+1]
             + la[0][tx*bank_shift+2]+la[0][tx*bank_shift+3]
             + la[0][tx*bank_shift+4]+la[0][tx*bank_shift+5]
             + la[0][tx*bank_shift+6]+la[0][tx*bank_shift+7];
        WC[0+lda*(blkc)] =  res1;
    }
}


/**************************************************************
 *
 */

__global__ void
magmablas_chemv_200_L_update_mgpu_offset_32(
    int n, magmaFloatComplex alpha,
    magmaFloatComplex *A, int lda,
    magmaFloatComplex *x, int incx,
    magmaFloatComplex beta,
    magmaFloatComplex *y, int incy,
    magmaFloatComplex *WC,
    int my_gpu_id,
    int num_gpus,
    int nb,
    int kstan )
{
    int i;
    int tx  = threadIdx.x;
    int ind = blockIdx.x * chemv_bs + tx;
    magmaFloatComplex Ca;

    MAGMA_C_SET2REAL(Ca, 0);
    WC += ind + lda * blockIdx.x;

    for(i = blockIdx.x* chemv_bs; i < n; i += chemv_bs) {
        Ca += WC[0];
        WC += chemv_bs;
    }
    if( ind < n && ind >= kstan)
        y[ind * incy] = beta * y[ind * incy]  + alpha * Ca;
}


extern "C"
void magmablas_chemv_200_L_mgpu_offset_32(
    magma_int_t m, magmaFloatComplex alpha,
    magmaFloatComplex *A, magma_int_t lda,
    magmaFloatComplex *X, magma_int_t incx,
    magmaFloatComplex beta,
    magmaFloatComplex *Y, magma_int_t incy,
    magmaFloatComplex *dC_work,
    magma_int_t my_gpu_id,
    magma_int_t num_gpus,
    magma_int_t nb,
    magma_int_t offset,
    magma_int_t num_blocks_skipped)
{
    magma_int_t the_chosen_block_id = offset / nb;
    
    magma_int_t kstan = offset % nb;

    A += lda * num_blocks_skipped * nb + the_chosen_block_id * nb;
    X += the_chosen_block_id * nb;
    Y += the_chosen_block_id * nb;

    magma_int_t blocks;

    if (m % chemv_bs == 0)
        blocks = m / chemv_bs;
    else
        blocks = m / chemv_bs + 1;

    blocks -= the_chosen_block_id;

    dim3 grid(blocks, 1, 1);

    dim3 threads(nb, 8, 1);
    dim3 threads_u(nb, 1, 1);

    /*
     * If matrix size is multiple of chemv_bs, we use a specific code.
     * otherwise, we call the generic case.
     */
    if(m % chemv_bs == 0 )
    {
        magmablas_chemv_200_L_special_mgpu_offset_32 <<< grid, threads, 0, magma_stream >>>(
            m, alpha, A, lda, X, incx, beta, Y, incy, dC_work, my_gpu_id, num_gpus, nb, kstan);
    }
    else
    {
        magma_int_t m_mod_nb = m%chemv_bs - 1;

        magmablas_chemv_200_L_generic_mgpu_offset_32 <<< grid, threads, 0, magma_stream >>> (
            m, alpha, A, lda, X, incx ,beta, Y, incy, dC_work, m_mod_nb, my_gpu_id, num_gpus, nb, kstan);
    }

    magmablas_chemv_200_L_update_mgpu_offset_32<<< grid, threads_u, 0, magma_stream >>>(
        m, alpha, A, lda, X, incx, beta, Y, incy, dC_work, my_gpu_id, num_gpus, nb, kstan);
}

/*******************************************************************************
 *     Functions for each specific cases - Upper case
 */

__global__ void
magmablas_chemv_200_U_special_mgpu_offset_32(
    int n, magmaFloatComplex alpha,
    magmaFloatComplex *A, int lda,
    magmaFloatComplex *x, int incx,
    magmaFloatComplex  beta,
    magmaFloatComplex *y, int incy,
    magmaFloatComplex *WC,
    int my_gpu_id,
    int num_gpus,
    int nb,
    int kstan)
{
    int tx   = threadIdx.x;
    int ty   = threadIdx.y;
    int blkc = blockIdx.x;

    magmaFloatComplex res  = MAGMA_C_ZERO; // used in scan the row
    magmaFloatComplex res_ = MAGMA_C_ZERO; // used in scan the column
    magmaFloatComplex res1 = MAGMA_C_ZERO; // tem for res
    magmaFloatComplex res2 = MAGMA_C_ZERO; // tem for res_

    __shared__ magmaFloatComplex la   [chemv_bs][bank_shift];
    __shared__ magmaFloatComplex buff [chemv_bs];
    __shared__ magmaFloatComplex buff2 [chemv_bs];

    int break_d   =  chemv_bs * blkc;

    x  += (break_d + tx ) * incx;
    A  +=  break_d;
    A  +=  ty * lda + tx;

    if( ty == 0 )
    {
        buff[tx] = x[0];
        if(blkc == 0  && tx < kstan)
        {
            MAGMA_C_SET2REAL(buff[tx], 0.0);
        }
    } // obtain the vector x store in buff;

    if ( (blkc % num_gpus) == my_gpu_id)
    {
        A += lda * (blkc/num_gpus) * chemv_bs; // change

        #pragma unroll
        for(int j =0; j < chemv_bs; j += 8)
            la[0][ bank_shift * (ty+j) + tx] =  A[ j * lda];
        __syncthreads();

        #pragma unroll
        for(int  i=ty*4; i < (ty * 4 + 4); i++) {
            if ( i > tx )
            {
                la[0][bank_shift * tx + i] = cuConjf(la[0][ i * bank_shift + tx]);
            }
        }
        __syncthreads();

        #pragma unroll
        for(int j=0; j < 4; j++)
            res += cuConjf( la[0][bank_shift * tx + j + ty * 4] ) * buff[j + ty * 4];
            
        __syncthreads();

        A -= lda * (blkc/num_gpus) * chemv_bs;
    }
    __syncthreads();

    x -= (break_d + tx ) * incx; // return to the beginning

    x += (my_gpu_id ) * chemv_bs; //

    int wc_c = my_gpu_id;

    int total_blocks_gpu = gridDim.x /num_gpus;

    if( my_gpu_id < ( gridDim.x % num_gpus) )
    {
        total_blocks_gpu += 1;
    }

    int shift = (blkc +1) /num_gpus;
    
    if( my_gpu_id < ( (blkc+1) % num_gpus) )
    {
        shift += 1;
    }

    #pragma unroll
    for(int s=0; s < shift; s++)
    {
        x += num_gpus * chemv_bs;
        A += lda * chemv_bs;
        wc_c += num_gpus;
    }

    WC +=  break_d + tx;
   
    int num_blocks_iters = total_blocks_gpu - shift;

    int count = 0;

    for(int s=0; s < num_blocks_iters; s++)
    {
        MAGMA_C_SET2REAL(res_,0);
        count++;

        #pragma unroll
        for(int j =0; j < chemv_bs; j += 8)
            la[0][ bank_shift * (ty+j) + tx] =  A[ j * lda];

        if( ty == 0 )
        {
            buff2[tx] = x[tx];
        } // obtain the vector x store in buff;
        __syncthreads();

        #pragma unroll
        for(int j=0; j < 4; j++)
        {
            res += (la[0][bank_shift * (ty + j * 8) + tx] )* buff2[ ty + j * 8];
            res_ += cuConjf( la[0][bank_shift * tx + j + ty * 4] ) * buff[j + ty * 4]; //iterate colum
        }
        __syncthreads();

        la[0][bank_shift*tx+ty]= res_;
        __syncthreads();

        if( ty == 0 )
        {
            res2 = la[0][tx*bank_shift+0]+la[0][tx*bank_shift+1]
                 + la[0][tx*bank_shift+2]+la[0][tx*bank_shift+3]
                 + la[0][tx*bank_shift+4]+la[0][tx*bank_shift+5]
                 + la[0][tx*bank_shift+6]+la[0][tx*bank_shift+7];
            
            WC[wc_c*lda ] = res2;
        }
        __syncthreads();

        wc_c += num_gpus;
        x += num_gpus * chemv_bs;
        A += lda * chemv_bs;
    }

    la[0][bank_shift*tx+ty]= res;
    __syncthreads();

    if( ty == 0 )
    {
        res1 = la[0][tx*bank_shift+0]+la[0][tx*bank_shift+1]
             + la[0][tx*bank_shift+2]+la[0][tx*bank_shift+3]
             + la[0][tx*bank_shift+4]+la[0][tx*bank_shift+5]
             + la[0][tx*bank_shift+6]+la[0][tx*bank_shift+7];
          
        WC[0+lda*(blkc)] =  res1;
    }
}


__global__ void
magmablas_chemv_200_U_generic_mgpu_offset_32(
    int n, magmaFloatComplex alpha,
    magmaFloatComplex *A, int lda,
    magmaFloatComplex *x, int incx,
    magmaFloatComplex beta,
    magmaFloatComplex *y, int incy,
    magmaFloatComplex *WC,
    int m_mod_thread_x,
    int my_gpu_id,
    int num_gpus,
    int nb,
    int kstan,
    int the_right_gpu)
{
    int tx   = threadIdx.x;
    int ty   = threadIdx.y;
    int blkc = blockIdx.x;

    magmaFloatComplex res  = MAGMA_C_ZERO;
    magmaFloatComplex res_ = MAGMA_C_ZERO;
    magmaFloatComplex res1 = MAGMA_C_ZERO;
    magmaFloatComplex res2 = MAGMA_C_ZERO;

    __shared__ magmaFloatComplex la   [chemv_bs][bank_shift];
    __shared__ magmaFloatComplex buff [chemv_bs];
    __shared__ magmaFloatComplex buff2 [chemv_bs];

    int break_d   =  chemv_bs * blkc;

    x += (break_d + tx ) * incx;
    A +=  break_d;
    A += lda * ty;

    int trackA;
    if( blkc == ( gridDim.x - 1 ))
    {
        if( ty == 0 ) {
            if( tx > m_mod_thread_x )
            {
                MAGMA_C_SET2REAL(buff[tx],0);
            }
            else
                buff[tx]  = x[0];
        }
        if ( tx > m_mod_thread_x )
            trackA=m_mod_thread_x;
        else
            trackA=tx;
        A += trackA;
    }
    else
    {
        if( ty == 0 )
        {
            buff[tx]  = x[0];
        }
        
        A += tx;
    }

    if(ty == 0 )
    {
        if(blkc == 0  && tx < kstan)//
        {
            MAGMA_C_SET2REAL(buff[tx], 0.0);
        }
    }
     
    if ( (blkc % num_gpus) == my_gpu_id)
    {
        A += lda * (blkc/num_gpus) * chemv_bs; // change

        if( blkc == ( gridDim.x - 1 ) ) {
            #pragma unroll
            for(int j =0; j < chemv_bs; j += 8) {
                if( ( ty + j ) > m_mod_thread_x )
                {
                    MAGMA_C_SET2REAL(la[0][bank_shift*(ty+j)+tx], 9999);
                }
                else
                    la[0][bank_shift*(ty+j)+tx] =  A[ j * lda];
            }
        }
        else {
            #pragma unroll
            for(int j =0; j < chemv_bs; j += 8) {
                la[0][bank_shift*(ty+j)+tx] = A[ j * lda];
            }
        }
        __syncthreads();

        #pragma unroll
        for(int  i=ty*4; i < (ty*4+4); i++) {
            if ( i > tx )
            {
                la[0][bank_shift * tx + i] = cuConjf(la[0][ i * bank_shift + tx]);
            }
        }
        __syncthreads();

        #pragma unroll
        for(int j=0; j < 4; j++)
            res += cuConjf(la[0][bank_shift*tx+j+ty*4])* buff[j+ty*4];
        __syncthreads();

        A -= lda * (blkc/num_gpus) * chemv_bs;
    }

    x  -= (break_d + tx ) * incx; // return to the beginning

    x += (my_gpu_id ) * chemv_bs; //

    int wc_c = my_gpu_id;

    int total_blocks_gpu = gridDim.x /num_gpus;

    if( my_gpu_id < ( gridDim.x % num_gpus) )
    {
        total_blocks_gpu += 1;
    }

    int shift = (blkc +1) /num_gpus;
        
    if( my_gpu_id < ( (blkc+1) % num_gpus) )
    {
        shift += 1;
    }

    #pragma unroll
    for(int s=0; s < shift; s++)
    {
        x += num_gpus * chemv_bs;
        A += lda * chemv_bs;
        wc_c += num_gpus;
    }
    
    WC +=  break_d + tx;
    
    int num_blocks_iters = total_blocks_gpu - shift;

    int count = 0;

    for(int s=0; s < num_blocks_iters; s++)
    {
        MAGMA_C_SET2REAL(res_,0);
        count++;
        
        if(my_gpu_id == the_right_gpu && s == num_blocks_iters-1)
        {
            if( ty == 0 )
            {
                if( tx > m_mod_thread_x )
                {
                    MAGMA_C_SET2REAL(buff2[tx],0);
                }
                else
                    buff2[tx]  = x[tx];
            }
            
            #pragma unroll
            for(int j =0; j < chemv_bs; j += 8)
            {
                if( ( ty + j ) > m_mod_thread_x )
                {
                    MAGMA_C_SET2REAL(la[0][bank_shift*(ty+j)+tx], 0);
                }
                else
                    la[0][bank_shift*(ty+j)+tx] =  A[ j * lda];
            }
            __syncthreads();
        } // end of the_right_gpu
        else
        {
            #pragma unroll
            for(int j =0; j < chemv_bs; j += 8)
                la[0][ bank_shift * (ty+j) + tx] =  A[ j * lda];
            
            if( ty == 0 )
            {
                buff2[tx] = x[tx];
            } // obtain the vector x store in buff;
            __syncthreads();
        }

        #pragma unroll
        for(int j=0; j < 4; j++)
        {
            res += (la[0][bank_shift * (ty + j * 8) + tx] )* buff2[ ty + j * 8];
            res_ += cuConjf( la[0][bank_shift * tx + j + ty * 4] ) * buff[j + ty * 4]; //iterate colum
        }
        __syncthreads();
    
        la[0][bank_shift*tx+ty]= res_;
        __syncthreads();
    
        if( ty == 0 )
        {
            res2 = la[0][tx*bank_shift+0]+la[0][tx*bank_shift+1]
                 + la[0][tx*bank_shift+2]+la[0][tx*bank_shift+3]
                 + la[0][tx*bank_shift+4]+la[0][tx*bank_shift+5]
                 + la[0][tx*bank_shift+6]+la[0][tx*bank_shift+7];
            
            WC[wc_c*lda ] = res2;
        }
        __syncthreads();
    
        wc_c += num_gpus;
        x += num_gpus * chemv_bs;
        A += lda * chemv_bs;
    }

    la[0][bank_shift*tx+ty]= res;
    __syncthreads();

    if( ty == 0 )
    {
        res1 = la[0][tx*bank_shift+0]+la[0][tx*bank_shift+1]
             + la[0][tx*bank_shift+2]+la[0][tx*bank_shift+3]
             + la[0][tx*bank_shift+4]+la[0][tx*bank_shift+5]
             + la[0][tx*bank_shift+6]+la[0][tx*bank_shift+7];
        WC[0+lda*(blkc)] =  res1;
    }
}


__global__ void
magmablas_chemv_200_U_update_mgpu_offset_32(
    int n, magmaFloatComplex alpha,
    magmaFloatComplex *A, int lda,
    magmaFloatComplex *x, int incx,
    magmaFloatComplex beta,
    magmaFloatComplex *y, int incy,
    magmaFloatComplex *WC,
    int my_gpu_id,
    int num_gpus,
    int nb,
    int kstan )
{
    int i;
    int tx  = threadIdx.x;
    int ind = blockIdx.x * chemv_bs + tx;
    magmaFloatComplex Ca;

    MAGMA_C_SET2REAL(Ca, 0);
    WC+=  blockIdx.x * lda + tx;

    for(i = 0; i < (blockIdx.x+1)*chemv_bs; i += chemv_bs)
    {
        Ca += WC[0];
        WC += chemv_bs;
    }
    if( ind < n && ind >= kstan)
        y[ind * incy] = beta * y[ind * incy]  + alpha * Ca;
}


extern "C"
void magmablas_chemv_200_U_mgpu_offset_32(
    magma_int_t m, magmaFloatComplex alpha,
    magmaFloatComplex *A, magma_int_t lda,
    magmaFloatComplex *X, magma_int_t incx,
    magmaFloatComplex beta,
    magmaFloatComplex *Y, magma_int_t incy,
    magmaFloatComplex *dC_work,
    magma_int_t my_gpu_id,
    magma_int_t num_gpus,
    magma_int_t nb,
    magma_int_t offset,
    magma_int_t num_blocks_skipped,
    magma_int_t the_right_gpu)
{
    magma_int_t the_chosen_block_id = offset / nb;
    magma_int_t kstan = offset % nb;

    A += lda * num_blocks_skipped * nb + the_chosen_block_id * nb;
    X += the_chosen_block_id * nb;
    Y += the_chosen_block_id * nb;

    magma_int_t blocks;

    if (m % chemv_bs == 0)
        blocks = m / chemv_bs;
    else
        blocks = m / chemv_bs + 1;

    blocks -= the_chosen_block_id;

    dim3 grid(blocks, 1, 1);
    dim3 threads(nb, 8, 1);
    dim3 threads_u(nb, 1, 1);

    /*
     * If matrix size is multiple of chemv_bs, we use a specific code.
     * otherwise, we call the generic case.
     */
    if(m % chemv_bs == 0 ) {
        magmablas_chemv_200_U_special_mgpu_offset_32 <<< grid, threads, 0, magma_stream >>>(
            m, alpha, A, lda, X, incx, beta, Y, incy, dC_work, my_gpu_id, num_gpus, nb, kstan);
    }
    else {
        magma_int_t m_mod_thread_x = m%chemv_bs - 1;

        magmablas_chemv_200_U_generic_mgpu_offset_32 <<< grid, threads, 0, magma_stream >>> (
            m, alpha, A, lda, X, incx ,beta, Y, incy, dC_work, m_mod_thread_x, my_gpu_id, num_gpus, nb, kstan, the_right_gpu);
    }

    magmablas_chemv_200_U_update_mgpu_offset_32<<< grid, threads_u, 0, magma_stream >>>(
        m, alpha, A, lda, X, incx, beta, Y, incy, dC_work, my_gpu_id, num_gpus, nb, kstan);
}


/*************************************************************************

    Purpose
    =======

    magmablas_chemv  performs the matrix-vector operation on fermi:

       y := alpha*A*x + beta*y,

    where alpha and beta are scalars, x and y are n element vectors and
    A is an n by n hermitian matrix.

    Arguments
    ==========

    UPLO   - CHARACTER*1.
             On entry, UPLO specifies whether the upper or lower
             triangular part of the array A is to be referenced as
             follows:

                UPLO = 'U' or 'u'   Only the upper triangular part of A
                                    is to be referenced.

                UPLO = 'L' or 'l'   Only the lower triangular part of A
                                    is to be referenced.

             Unchanged on exit.

    N      - INTEGER.
             On entry, N specifies the order of the matrix A.
             N must be at least zero.
             Unchanged on exit.

    ALPHA  - COMPLEX*16      .
             On entry, ALPHA specifies the scalar alpha.
             Unchanged on exit.

    A      - COMPLEX*16       array of DIMENSION ( LDA, n ).
             Before entry with  UPLO = 'U' or 'u', the leading n by n
             upper triangular part of the array A must contain the upper
             triangular part of the hermitian matrix and the strictly
             lower triangular part of A is not referenced.
             Before entry with UPLO = 'L' or 'l', the leading n by n
             lower triangular part of the array A must contain the lower
             triangular part of the hermitian matrix and the strictly
             upper triangular part of A is not referenced.
             Note that the imaginary parts of the diagonal elements need
             not be set and are assumed to be zero.
             Unchanged on exit.

    LDA    - INTEGER.
             On entry, LDA specifies the first dimension of A as declared
             in the calling (sub) program. LDA must be at least
             max( 1, n ).
             Unchanged on exit.
             It is recommended that lda is multiple of 16. Otherwise
             performance would be deteriorated as the memory accesses
             would not be fully coalescent.

    X      - COMPLEX*16       array of dimension at least
             ( 1 + ( n - 1 )*abs( INCX ) ).
             Before entry, the incremented array X must contain the n
             element vector x.
             Unchanged on exit.

    INCX   - INTEGER.
             On entry, INCX specifies the increment for the elements of
             X. INCX must not be zero.
             Unchanged on exit.

    BETA   - COMPLEX*16      .
             On entry, BETA specifies the scalar beta. When BETA is
             supplied as zero then Y need not be set on input.
             Unchanged on exit.

    Y      - COMPLEX*16       array of dimension at least
             ( 1 + ( n - 1 )*abs( INCY ) ).
             Before entry, the incremented array Y must contain the n
             element vector y. On exit, Y is overwritten by the updated
             vector y.

    INCY   - INTEGER.
             On entry, INCY specifies the increment for the elements of
             Y. INCY must not be zero.
             Unchanged on exit.

*/


extern "C"
magma_int_t
magmablas_chemv_mgpu_32_offset(
    char uplo, magma_int_t n,
    magmaFloatComplex alpha,
    magmaFloatComplex **A, magma_int_t lda,
    magmaFloatComplex **X, magma_int_t incx,
    magmaFloatComplex beta,
    magmaFloatComplex **Y, magma_int_t incy,
    magmaFloatComplex **work, magma_int_t lwork,
    magma_int_t num_gpus,
    magma_int_t nb,
    magma_int_t offset,
    magma_queue_t stream[][10])
{
    char uplo_[2] = {uplo, 0};
    int  upper    = lapackf77_lsame(uplo_, "U");

    /*
     * Test the input parameters.
     */
    
    if ((! upper) && (! lapackf77_lsame(uplo_, "L"))) {
        return -1;
    } else if ( n < 0 ) {
        return -2;
    } else if ( lda < max(1,n) ) {
        return -5;
    } else if ( incx == 0 ) {
        return -7;
    } else if ( incy == 0 ) {
        return -10;
    }

    /*
     * Quick return if possible.
     */
    if ( (n == 0) || ( MAGMA_C_EQUAL(alpha, MAGMA_C_ZERO) && MAGMA_C_EQUAL(beta, MAGMA_C_ONE) ) )
        return MAGMA_SUCCESS;

    magma_int_t blocks    = n / chemv_bs + (n % chemv_bs != 0);
    magma_int_t workspace = lda * (blocks + 1);

    if (lwork < workspace) {
        printf("Not enough work space in magmablas_chemv: passed %d, required %d\n",
               (int) lwork, (int) workspace);
        exit(1);
    }
    if(nb != 32)
    {
        printf("Error in magmablas_chemv_200_mgpu: nb != 32, program will exit! please reallocate your matrix among GPUs\n");
        exit(0);
    }
    magma_int_t i = 0;
    for(i=0; i < num_gpus; i++)
    {
        magma_setdevice(i);
        magmablasSetKernelStream(stream[i][0]);

        magma_int_t the_chosen_block_id = offset / nb;
        magma_int_t the_chosen_gpu_id = the_chosen_block_id % num_gpus;

        magma_int_t  num_blocks_skipped = the_chosen_block_id / num_gpus;
     
        if(i < the_chosen_gpu_id)
        {
            num_blocks_skipped += 1;
        }
        
        int new_gpu_id = ( i + num_gpus - the_chosen_gpu_id ) % num_gpus;

        magma_int_t the_right_block_id = n / nb;
        magma_int_t the_right_gpu = the_right_block_id % num_gpus;

        the_right_gpu = ( the_right_gpu + num_gpus - the_chosen_gpu_id ) % num_gpus;
        // the_right_gpu is used in Upper generic case.

        if ( upper)
        {
            magmablas_chemv_200_U_mgpu_offset_32(n, alpha, A[i], lda, X[i], incx, beta, Y[i], incy, work[i],
                             new_gpu_id, num_gpus, nb, offset, num_blocks_skipped, the_right_gpu);
        }
        else
        {
            magmablas_chemv_200_L_mgpu_offset_32(n, alpha, A[i], lda, X[i], incx, beta, Y[i], incy, work[i],
                                                   new_gpu_id, num_gpus, nb, offset, num_blocks_skipped);
        }
    }

    return MAGMA_SUCCESS;
}


extern "C"
magma_int_t
magmablas_chemv2_mgpu_32_offset(
    char uplo, magma_int_t n,
    magmaFloatComplex alpha,
    magmaFloatComplex **A, magma_int_t lda,
    magmaFloatComplex **X, magma_int_t incx,
    magmaFloatComplex beta,
    magmaFloatComplex **Y, magma_int_t incy,
    magmaFloatComplex **work, magma_int_t lwork,
    magma_int_t num_gpus,
    magma_int_t nb,
    magma_int_t offset)
{
    char uplo_[2] = {uplo, 0};
    int  upper    = lapackf77_lsame(uplo_, "U");

    /*
     * Test the input parameters.
     */
    if ((! upper) && (! lapackf77_lsame(uplo_, "L"))) {
        return -1;
    } else if ( n < 0 ) {
        return -2;
    } else if ( lda < max(1,n) ) {
        return -5;
    } else if ( incx == 0 ) {
        return -7;
    } else if ( incy == 0 ) {
        return -10;
    }

    /*
     * Quick return if possible.
     */
    if ( (n == 0) || ( MAGMA_C_EQUAL(alpha, MAGMA_C_ZERO) && MAGMA_C_EQUAL(beta, MAGMA_C_ONE) ) )
        return MAGMA_SUCCESS;

    magma_int_t blocks    = n / chemv_bs + (n % chemv_bs != 0);
    magma_int_t workspace = lda * (blocks + 1);

    if (lwork < workspace) {
        printf("Not enough work space in magmablas_chemv: passed %d, required %d\n",
               (int) lwork, (int) workspace);
        exit(1);
    }
    if(nb != 32)
    {
        printf("Error in magmablas_chemv_200_mgpu: nb != 32, program will exit! please reallocate your matrix among GPUs\n");
        exit(0);
    }
    magma_int_t i = 0;
    for(i=0; i < num_gpus; i++)
    {
        magma_setdevice(i);
        // magmablasSetKernelStream(stream[i][0]);

        magma_int_t the_chosen_block_id = offset / nb;
        magma_int_t the_chosen_gpu_id = the_chosen_block_id % num_gpus;

        magma_int_t  num_blocks_skipped = the_chosen_block_id / num_gpus;

        if(i < the_chosen_gpu_id)
        {
            num_blocks_skipped += 1;
        }
         
        int new_gpu_id = ( i + num_gpus - the_chosen_gpu_id ) % num_gpus;
         
        magma_int_t the_right_block_id = n / nb;
        magma_int_t the_right_gpu = the_right_block_id % num_gpus;

        the_right_gpu = ( the_right_gpu + num_gpus - the_chosen_gpu_id ) % num_gpus;
        // the_right_gpu is used in Upper generic case.

        if ( upper)
        {
            magmablas_chemv_200_U_mgpu_offset_32(n, alpha, A[i], lda, X[i], incx, beta, Y[i], incy, work[i],
                new_gpu_id, num_gpus, nb, offset, num_blocks_skipped, the_right_gpu);
        }
        else
            magmablas_chemv_200_L_mgpu_offset_32(n, alpha, A[i], lda, X[i], incx, beta, Y[i], incy, work[i],
                new_gpu_id, num_gpus, nb, offset, num_blocks_skipped);
    }

    return MAGMA_SUCCESS;
}


extern "C"
magma_int_t
magmablas_chemv2_mgpu_32(
    char uplo, magma_int_t n,
    magmaFloatComplex alpha,
    magmaFloatComplex **A, magma_int_t lda,
    magmaFloatComplex **X, magma_int_t incx,
    magmaFloatComplex beta,
    magmaFloatComplex **Y, magma_int_t incy,
    magmaFloatComplex **work, magma_int_t lwork,
    magma_int_t num_gpus,
    magma_int_t nb)
{
    char uplo_[2] = {uplo, 0};
    int  upper    = lapackf77_lsame(uplo_, "U");

    /*
     * Test the input parameters.
     */
    if ((! upper) && (! lapackf77_lsame(uplo_, "L"))) {
        return -1;
    } else if ( n < 0 ) {
        return -2;
    } else if ( lda < max(1,n) ) {
        return -5;
    } else if ( incx == 0 ) {
        return -7;
    } else if ( incy == 0 ) {
        return -10;
    }

    /*
     * Quick return if possible.
     */
    if ( (n == 0) || ( MAGMA_C_EQUAL(alpha, MAGMA_C_ZERO) && MAGMA_C_EQUAL(beta, MAGMA_C_ONE) ) )
        return MAGMA_SUCCESS;

    magma_int_t blocks    = n / chemv_bs + (n % chemv_bs != 0);
    magma_int_t workspace = lda * (blocks + 1);

    if (lwork < workspace) {
        printf("Not enough work space in magmablas_chemv: passed %d, required %d\n",
               (int) lwork, (int) workspace);
        exit(1);
    }
    if(nb != 32)
    {
        printf("Error in magmablas_chemv_200_mgpu: nb != 32, program will exit! please reallocate your matrix among GPUs\n");
        exit(0);
    }
    magma_int_t i = 0;

    for(i=0; i < num_gpus; i++)
    {
        magma_setdevice(i);
         
        magma_int_t the_right_block_id = n / nb;
        magma_int_t the_right_gpu = the_right_block_id % num_gpus;

        // the_right_gpu is used in Upper generic case.

        if ( upper)
        {
            magmablas_chemv_200_U_mgpu_offset_32(n, alpha, A[i], lda, X[i], incx, beta, Y[i], incy, work[i],
                i, num_gpus, nb, 0, 0, the_right_gpu);
        }
        else
            magmablas_chemv_200_L_mgpu_offset_32(n, alpha, A[i], lda, X[i], incx, beta, Y[i], incy, work[i],
                i, num_gpus, nb, 0, 0);
    }

    return MAGMA_SUCCESS;
}

#endif
