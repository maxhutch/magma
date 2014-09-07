/*
    -- MAGMA (version 1.5.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date September 2014

       @precisions normal z -> s d c

*/
#include "common_magma.h"

#define PRECISION_z

#define hemv_bs          64
#define thread_x         64
#define thread_y          4
#define bank_shift       33
#define quarter_thread_x 16
#define half_thread_x    32

/*******************************************************************************
 *    Lower case, where n is multiple of block size (hemv_bs)
 */

__global__ void
zhemv_kernel_fermi_L_special_mgpu_offset(
    int n, magmaDoubleComplex alpha,
    magmaDoubleComplex *A, int lda,
    magmaDoubleComplex *x, int incx,
    magmaDoubleComplex  beta,
    magmaDoubleComplex *y, int incy,
    magmaDoubleComplex *WC,
    int my_gpu_id,
    int num_gpus,
    int nb,
    int the_chosen_block_id,
    int the_chosen_gpu_id,
    int kstan)
{
#if (__CUDA_ARCH__ >= 200)
    int tx   = threadIdx.x;
    int ty   = threadIdx.y;
    int blkc = blockIdx.x;

    if ( blkc < my_gpu_id ) {
        return;
    }

    magmaDoubleComplex res  = MAGMA_Z_ZERO;
    magmaDoubleComplex res_ = MAGMA_Z_ZERO;
    magmaDoubleComplex res1 = MAGMA_Z_ZERO;
    magmaDoubleComplex res2 = MAGMA_Z_ZERO;

    __shared__ magmaDoubleComplex la   [quarter_thread_x][thread_x+2];
    __shared__ magmaDoubleComplex buff [thread_x];
    __shared__ magmaDoubleComplex buff2 [thread_x];

    magmaDoubleComplex tr[4];
    magmaDoubleComplex b[4];

    int break_d   =  thread_x * blkc;
    const int td  = (thread_x * ty) + tx;
    int       tx_ = td % half_thread_x;
    int       ty_ = td / half_thread_x;

    WC +=  break_d + tx;
    x  += (break_d + tx) * incx;
    A  +=  break_d;
    A  += ty_ * lda + tx_;

    if ( ty == 0 ) {
        buff[tx] = x[0];
        if ( blkc == the_chosen_block_id && my_gpu_id == the_chosen_gpu_id && tx < kstan ) {
            buff[tx] = MAGMA_Z_ZERO;
        }
    } // obtain the vector x store in buff;
    // __syncthreads();

    tx = tx_; ty = ty_;

    int flag = 0;

    if ( (blkc % num_gpus) == my_gpu_id ) {
        A += lda * (blkc/num_gpus) * thread_x; // change

        #pragma unroll
        for(int j=0; j < half_thread_x; j += 8)
            la[0][ bank_shift * (ty_+j) + tx_] = A[ j * lda];
        __syncthreads();

        #pragma unroll
        for(int i=ty_*4; i < (ty_ * 4 + 4); i++) {
            if ( i < tx_ ) {
                la[0][bank_shift * tx_ + i] = cuConj( la[0][ i * bank_shift + tx_] );
            }
            else
                la[0][bank_shift * tx_ + i] = la[0][ bank_shift * tx_ + i];
        }
        __syncthreads();

        #pragma unroll
        for(int j=0; j < 4; j++)
            res += cuConj( la[0][bank_shift * tx_ + j + ty_ * 4] ) * buff[j + ty_ * 4];
        __syncthreads();
        
        la[0][bank_shift*tx_+ty_] = res;
        __syncthreads();
        
        if ( ty_ == 0 )
            res1 = la[0][tx_*bank_shift+0]+la[0][tx_*bank_shift+1]
                 + la[0][tx_*bank_shift+2]+la[0][tx_*bank_shift+3]
                 + la[0][tx_*bank_shift+4]+la[0][tx_*bank_shift+5]
                 + la[0][tx_*bank_shift+6]+la[0][tx_*bank_shift+7];
        else {
            res1 = MAGMA_Z_ZERO;
        }
        __syncthreads();
        
        res = MAGMA_Z_ZERO;
        
        A += half_thread_x + half_thread_x*lda;
        
        #pragma unroll
        for(int j=0; j < half_thread_x; j += 8)
            la[0][bank_shift*(ty_+j)+tx_] = A[ j * lda];
        __syncthreads();
        
        #pragma unroll
        for(int i=ty_*4; i < (4+ty_*4); i++) {
            if ( i < tx_ ) {
                la[0][bank_shift*tx_+i] = cuConj( la[0][bank_shift*i+tx_] );
            }
            else
                la[0][bank_shift*tx_+i] = la[0][bank_shift*tx_+i];
        }
        __syncthreads();
        
        #pragma unroll
        for(int j=0; j < 4; j++)
            res += cuConj( la[0][bank_shift*tx_+j+ty_*4] ) * buff[half_thread_x + j + 4 * ty_];
        __syncthreads();
        la[0][bank_shift*tx_+ty_] = res;
        __syncthreads();
        
        if ( ty_ == 1 )
            res2 = la[0][tx_*bank_shift+0]+la[0][tx_*bank_shift+1]
                 + la[0][tx_*bank_shift+2]+la[0][tx_*bank_shift+3]
                 + la[0][tx_*bank_shift+4]+la[0][tx_*bank_shift+5]
                 + la[0][tx_*bank_shift+6]+la[0][tx_*bank_shift+7];
        else {
            res2 = MAGMA_Z_ZERO;
        }
        __syncthreads();
        
        res = MAGMA_Z_ZERO;
        
        A -= half_thread_x*lda;
        
        res_ = MAGMA_Z_ZERO;
        
        #pragma unroll
        for(int j=0; j < half_thread_x; j += 8)
            tr[j/8] = A[ j * lda];
        
        #pragma unroll
        for(int j=0; j < 4; j++) {
            res += tr[j] * buff[ j*8 + ty_];
            la[0][bank_shift*(ty_+j*8)+tx_] = tr[j];
        }
        __syncthreads();
        
        #pragma unroll
        for(int j=0; j < 4; j++)
            res_ += cuConj(la[0][bank_shift*tx_+j+ty_*4]) * buff[half_thread_x +j+ty_*4];
        __syncthreads();
        
        la[0][bank_shift*tx_+ty_] = res;
        __syncthreads();
        if ( ty_ == 1 )
            res2 = res2
                 + la[0][tx_*bank_shift+0]+la[0][tx_*bank_shift+1]
                 + la[0][tx_*bank_shift+2]+la[0][tx_*bank_shift+3]
                 + la[0][tx_*bank_shift+4]+la[0][tx_*bank_shift+5]
                 + la[0][tx_*bank_shift+6]+la[0][tx_*bank_shift+7];
        else {
            res2 = MAGMA_Z_ZERO;
        }
        __syncthreads();
        
        la[0][bank_shift*tx_+ty_] = res_;
        __syncthreads();
        if ( ty_ == 0 ) {
            res1 = res1
                 + la[0][tx_*bank_shift+0]+la[0][tx_*bank_shift+1]
                 + la[0][tx_*bank_shift+2]+la[0][tx_*bank_shift+3]
                 + la[0][tx_*bank_shift+4]+la[0][tx_*bank_shift+5]
                 + la[0][tx_*bank_shift+6]+la[0][tx_*bank_shift+7];
        }
        else {
            res1 = MAGMA_Z_ZERO;
        }
        __syncthreads();
        A -= half_thread_x;
        
        flag = 1;
        
        A -= lda * (blkc/num_gpus) * thread_x;
    }

    tx = threadIdx.x;
    ty = threadIdx.y;

    if ( ty_ == 0  && ty == 0  )
        res = res1;
    else if ( ty_ == 1  && ty == 0  )
        res = res2;
    else {
        res = MAGMA_Z_ZERO;
    }

    A -= ty_ * lda;
    A -= tx_;

    x = x - blkc * thread_x * incx;

    //x = x - tx*incx;

    A += 4 * ty * lda;
    A += tx;

    int count = 0;

    tx_ = td % quarter_thread_x;
    ty_ = td / quarter_thread_x;

    WC -= tx;
    WC += tx_;

    int num_blocks_iters = (blkc +1) /num_gpus - flag;
    
    if ( my_gpu_id < ( (blkc+1) % num_gpus) ) {
        num_blocks_iters += 1;
    }

    x += (my_gpu_id) * nb;
    int wc_c = my_gpu_id;

    if ( blkc > my_gpu_id ) {
        for(int s=0; s < num_blocks_iters; s++) {
            res_ = MAGMA_Z_ZERO;
    
            count++;
    
            if ( ty == 0 ) {
                buff2[tx] = x[0];
                if ( my_gpu_id == the_chosen_gpu_id && tx < kstan && count == 1 ) {
                    buff2[tx] = MAGMA_Z_ZERO;
                }
            }
            __syncthreads();
    
            #pragma unroll
            for( int k=0; k < 4; k++ ) {
                #pragma unroll
                for(int j=0; j < 4; j++)
                    tr[j] = A[j*lda];
            
                #pragma unroll
                for(int j=0; j < 4; j++) {
                    res += tr[j] * buff2[ quarter_thread_x * k + ty*4 + j];
                    la[j + ty*4][tx] = cuConj(tr[j]) * buff[tx];
                    //la[j + ty*4][tx] = cuConj(tr[j]);
                }
                __syncthreads();
            
                res_ = MAGMA_Z_ZERO;
            
                #pragma unroll
                for(int j=0; j < 4; j++) {
                    //res_ += la[tx_][ty_*4+j] * b[j];
                    res_ += la[tx_][ty_*4+j];
                }
                //b[4 + k] = res_;
                b[ k] = res_;
                __syncthreads();
    
                A += lda * quarter_thread_x;
            }
    
            #pragma unroll
            for(int k=0; k < 4; k++) {
                //la[tx_][ty_+quarter_thread_x*k] = b[4+k];
                la[tx_][ty_+quarter_thread_x*k] = b[k];
            }
            __syncthreads();
            if ( ty_ < 4 ) {
                int k = ty_*quarter_thread_x;
                res_ = la[tx_][0+k] + la[tx_][1+k]
                     + la[tx_][2+k] + la[tx_][3+k]
                     + la[tx_][4+k] + la[tx_][5+k]
                     + la[tx_][6+k] + la[tx_][7+k]
                     + la[tx_][8+k] + la[tx_][9+k]
                     + la[tx_][10+k]+ la[tx_][11+k]
                     + la[tx_][12+k]+ la[tx_][13+k]
                     + la[tx_][14+k]+ la[tx_][15+k];
                WC[k + wc_c*lda ] =   res_;
            }
    
            wc_c += num_gpus;
            x += num_gpus * nb;
            __syncthreads();
        }
    }

    WC += tx;
    WC -= tx_;

    la[ty][tx] = res; // res store the swipe across the row
    __syncthreads();
    if ( ty == 0 ) {
        res = la[0][tx]+ la[1][tx]
            + la[2][tx]+ la[3][tx];
        WC[0+lda*(blkc)  ] =  res;
    }
#endif /* (__CUDA_ARCH__ >= 200) */
}

/**************************************************************
 *    Lower case for generic sizes
 */
__global__ void
zhemv_kernel_fermi_L_generic_mgpu_offset(
    int n, magmaDoubleComplex alpha,
    magmaDoubleComplex *A, int lda,
    magmaDoubleComplex *x, int incx,
    magmaDoubleComplex beta,
    magmaDoubleComplex *y, int incy,
    magmaDoubleComplex *WC,
    int m_mod_thread_x,
    int my_gpu_id,
    int num_gpus,
    int nb,
    int the_chosen_block_id,
    int the_chosen_gpu_id,
    int kstan)
{
#if (__CUDA_ARCH__ >= 200)
    int tx   = threadIdx.x;
    int ty   = threadIdx.y;
    int blkc = blockIdx.x;

    if ( blkc < my_gpu_id ) {
        return;
    }

    magmaDoubleComplex res  = MAGMA_Z_ZERO;
    magmaDoubleComplex res_ = MAGMA_Z_ZERO;
    magmaDoubleComplex res1 = MAGMA_Z_ZERO;
    magmaDoubleComplex res2 = MAGMA_Z_ZERO;

    __shared__ magmaDoubleComplex la   [quarter_thread_x][thread_x+2];
    __shared__ magmaDoubleComplex buff [thread_x];
    __shared__ magmaDoubleComplex buff2 [thread_x];

    magmaDoubleComplex tr[4];
    magmaDoubleComplex b[4];

    int break_d   =  thread_x * blkc;
    const int td  = (thread_x * ty) + tx;
    int       tx_ = td % half_thread_x;
    int       ty_ = td / half_thread_x;

    WC +=  break_d + tx;
    x  += (break_d + tx) * incx;
    A  +=  break_d;
    A  += lda * ty_;

    int trackA;
    if ( blkc == ( gridDim.x - 1 ) ) {
        if ( ty == 0 ) {
            if ( tx > m_mod_thread_x ) {
                buff[tx] = MAGMA_Z_ZERO;
            }
            else
                buff[tx]  = x[0];
        }
        if ( tx_ > m_mod_thread_x )
            trackA=m_mod_thread_x;
        else
            trackA=tx_;
        A += trackA;
    }
    else {
        if ( ty == 0 ) {
            buff[tx]  = x[0];
        }
        trackA = tx_;
        A += trackA;
    }

    if ( ty == 0 ) {
        if ( my_gpu_id == 0 && blkc == 0  && tx < kstan ) {
            buff[tx] = MAGMA_Z_ZERO;
        }
    }

    int flag = 0;
    
    if ( (blkc % num_gpus) == my_gpu_id ) {
        A += lda * (blkc/num_gpus) * thread_x; // change
        // Somehow merging these two if - else creates problem
        // It could be a potential bug -- from synchronization or from cuda or compiler
        if ( blkc == ( gridDim.x - 1 ) ) {
            #pragma unroll
            for(int j=0; j < half_thread_x; j += 8) {
                if ( ( ty_ + j ) > m_mod_thread_x ) {
                    la[0][bank_shift*(ty_+j)+tx_] = MAGMA_Z_MAKE( 9999, 0 );
                }
                else
                    la[0][bank_shift*(ty_+j)+tx_] = A[ j * lda];
            }
            A -= trackA;
        }
        else {
            #pragma unroll
            for(int j=0; j < half_thread_x; j += 8) {
                la[0][bank_shift*(ty_+j)+tx_] = A[ j * lda];
            }
        }
        tx = tx_;
        ty = ty_;
        __syncthreads();
        
        #pragma unroll
        for(int i=ty_*4; i < (ty_*4+4); i++) {
            if ( i < tx_ ) {
                la[0][bank_shift*tx_+i] = cuConj(la[0][i*bank_shift+tx_]);
            }
            else
                la[0][bank_shift*tx_+i] = la[0][bank_shift*tx_+i];
        }
        __syncthreads();
        
        #pragma unroll
        for(int j=0; j < 4; j++)
            res += cuConj(la[0][bank_shift*tx_+j+ty_*4])* buff[j+ty_*4];
        __syncthreads();
        
        la[0][bank_shift*tx_+ty_] = res;
        __syncthreads();
        if ( ty_ == 0 )
            res1 = la[0][tx_*bank_shift+0]
                 + la[0][tx_*bank_shift+1]
                 + la[0][tx_*bank_shift+2]
                 + la[0][tx_*bank_shift+3]
                 + la[0][tx_*bank_shift+4]
                 + la[0][tx_*bank_shift+5]
                 + la[0][tx_*bank_shift+6]
                 + la[0][tx_*bank_shift+7];
        else {
            res1 = MAGMA_Z_ZERO;
        }
        __syncthreads();
        
        res = MAGMA_Z_ZERO;
        
        if ( blkc == ( gridDim.x - 1 ) ) {
            if ( (tx_+half_thread_x) > m_mod_thread_x )
                trackA = m_mod_thread_x;
            else
                trackA = tx_ + half_thread_x;
            A += trackA+half_thread_x*lda;
        
            #pragma unroll
            for(int j=0; j < half_thread_x; j += 8) {
                if ( ( ty_ + j+half_thread_x ) > m_mod_thread_x ) {
                    la[0][bank_shift*(ty_+j)+tx_] = MAGMA_Z_MAKE( 99999, 0 );
                }
                else
                    la[0][bank_shift*(ty_+j)+tx_] = A[ j * lda];
            }
        
            A -= trackA+half_thread_x*lda;
            A += tx_;
            A += half_thread_x + half_thread_x*lda;
        }
        else {
            A += half_thread_x + half_thread_x*lda;
        
            #pragma unroll
            for(int j=0; j < half_thread_x; j += 8) {
                la[0][bank_shift*(ty_+j)+tx_] = A[ j * lda];
            }
        }
        
        __syncthreads();
        #pragma unroll
        for(int i=ty_*4; i < (4+ty_*4); i++) {
            if ( i < tx_ ) {
                la[0][bank_shift*tx_+i] = cuConj(la[0][bank_shift*i+tx_]);
            }
            else
                la[0][bank_shift*tx_+i] = la[0][bank_shift*tx_+i];
        }
        __syncthreads();
        
        #pragma unroll
        for(int j=0; j < 4; j++)
            res += cuConj(la[0][bank_shift*tx_+j+ty_*4]) * buff[half_thread_x + j + 4 * ty_];
        __syncthreads();
        
        la[0][bank_shift*tx_+ty_] = res;
        __syncthreads();
        
        if ( ty_ == 1 )
            res2 = la[0][tx_*bank_shift+0]
                 + la[0][tx_*bank_shift+1]
                 + la[0][tx_*bank_shift+2]
                 + la[0][tx_*bank_shift+3]
                 + la[0][tx_*bank_shift+4]
                 + la[0][tx_*bank_shift+5]
                 + la[0][tx_*bank_shift+6]
                 + la[0][tx_*bank_shift+7];
        else {
            res2 = MAGMA_Z_ZERO;
        }
        __syncthreads();
        
        res = MAGMA_Z_ZERO;
        res_ = MAGMA_Z_ZERO;
        
        A -= half_thread_x*lda;
        if ( blkc == ( gridDim.x - 1 ) ) {
            A -= tx_;
            if ( tx_ > m_mod_thread_x )
                trackA = m_mod_thread_x;
            else
                trackA = tx_;
            A += trackA;
        
            #pragma unroll
            for(int j=0; j < half_thread_x; j += 8)
                if ( ( ty_ + j ) > m_mod_thread_x ) {
                    tr[j/8] = MAGMA_Z_MAKE( 99999, 0 );
                }
                else
                    tr[j/8] = A[ j * lda];
            A -= trackA;
            A += tx_;
        }
        else {
            #pragma unroll
            for(int j=0; j < half_thread_x; j += 8)
                tr[j/8] = A[ j * lda];
        }
        __syncthreads();
        
        #pragma unroll
        for(int j=0; j < 4; j++) {
            res += tr[j] * buff[ j*8 + ty_];
            la[0][bank_shift*(ty_+j*8)+tx_] = tr[j];
        }
        __syncthreads();
        
        #pragma unroll
        for(int j=0; j < 4; j++)
            res_ += cuConj(la[0][bank_shift*tx_+j+ty_*4]) * buff[half_thread_x +j+ty_*4];
        __syncthreads();
        
        la[0][bank_shift*tx_+ty_] = res;
        __syncthreads();
        if ( ty_ == 1 )
            res2 = res2
                 + la[0][tx_*bank_shift+0]
                 + la[0][tx_*bank_shift+1]
                 + la[0][tx_*bank_shift+2]
                 + la[0][tx_*bank_shift+3]
                 + la[0][tx_*bank_shift+4]
                 + la[0][tx_*bank_shift+5]
                 + la[0][tx_*bank_shift+6]
                 + la[0][tx_*bank_shift+7];
        else {
            res2 = MAGMA_Z_ZERO;
        }
        __syncthreads();
        
        la[0][bank_shift*tx_+ty_] = res_;
        __syncthreads();
        
        if ( ty_ == 0 ) {
            res1 = res1
                 + la[0][tx_*bank_shift+0]
                 + la[0][tx_*bank_shift+1]
                 + la[0][tx_*bank_shift+2]
                 + la[0][tx_*bank_shift+3]
                 + la[0][tx_*bank_shift+4]
                 + la[0][tx_*bank_shift+5]
                 + la[0][tx_*bank_shift+6]
                 + la[0][tx_*bank_shift+7];
        }
        else {
            res1 = MAGMA_Z_ZERO;
        }
        A -= half_thread_x;

        A -= lda * (blkc/num_gpus) * thread_x;
        
        flag = 1;
    }

    __syncthreads();
    tx = threadIdx.x;
    ty = threadIdx.y;

    if ( ty_ == 0  && ty == 0  )
        res = res1;
    else if ( ty_ == 1  && ty == 0  )
        res = res2;
    else {
        res = MAGMA_Z_ZERO;
    }

    A -= ty_ * lda;
    A -= tx_;

    x = x - break_d*incx;
    //x = x - tx * incx;

    A += 4 * ty * lda;

    if ( blkc  == ( gridDim.x - 1 ) ) {
        if ( tx <= m_mod_thread_x )
            A += tx;
        else
            A += m_mod_thread_x;
    }
    else {
        A += tx;
    }

    int wc_c = my_gpu_id;
    int count = 0;

    tx_ = td % quarter_thread_x;
    ty_ = td / quarter_thread_x;

    WC -= tx;
    WC += tx_;

    int num_blocks_iters = (blkc +1) /num_gpus - flag;
    
    if ( my_gpu_id < ( (blkc+1) % num_gpus) ) {
        num_blocks_iters += 1;
    }

    x += (my_gpu_id) * nb;

    if ( blkc > my_gpu_id ) {
        for(int s=0; s < num_blocks_iters; s++) {
            res_ = MAGMA_Z_ZERO;
            count++;
            
            if ( ty == 0 ) {
                buff2[tx] = x[0];

                if ( my_gpu_id == the_chosen_gpu_id && tx < kstan && count == 1 ) {
                    buff2[tx] = MAGMA_Z_ZERO;
                }
            }
            __syncthreads();

            #pragma unroll
            for( int k=0; k < 4; k++ ) {
                #pragma unroll
                for(int j=0; j < 4; j++)
                    tr[j] = A[j*lda];

                #pragma unroll
                for(int j=0; j < 4; j++) {
                    res += tr[j] * buff2[ quarter_thread_x*k + ty*4+(j)];
                    la[( (j)+ty*4)][tx] = cuConj(tr[j]) * buff[tx];
                }
                __syncthreads();

                res_ = MAGMA_Z_ZERO;

                #pragma unroll
                for(int j=0; j < 4; j++)
                    res_ += la[tx_][ty_*4+j];
                b[k] = res_;
                __syncthreads();
                A += lda * quarter_thread_x;
            }

            #pragma unroll
            for(int k=0; k < 4; k++) {
                la[tx_][ty_+quarter_thread_x*k] = b[k];
            }
            __syncthreads();

            if ( ty_ < 4 ) {
                int k = ty_*quarter_thread_x;
                res_ = la[tx_][0+k] + la[tx_][1+k]
                     + la[tx_][2+k] + la[tx_][3+k]
                     + la[tx_][4+k] + la[tx_][5+k]
                     + la[tx_][6+k] + la[tx_][7+k]
                     + la[tx_][8+k] + la[tx_][9+k]
                     + la[tx_][10+k]+ la[tx_][11+k]
                     + la[tx_][12+k]+ la[tx_][13+k]
                     + la[tx_][14+k]+ la[tx_][15+k];
                WC[k + wc_c*lda ] =   res_;
            }
            
            wc_c += num_gpus;
            x += num_gpus * nb;

            __syncthreads();
        }
    }

    WC += tx;
    WC -= tx_;
    la[ty][tx] = res;
    __syncthreads();
        
    if ( ty == 0 ) {
        res=la[0][tx]+ la[1][tx]+ la[2][tx]+ la[3][tx];
        WC[0+lda*(blkc)] = res;
    }
#endif /* (__CUDA_ARCH__ >= 200) */
}

__global__ void
zhemv_kernel_fermi_L_update_mgpu_offset(
    int n, magmaDoubleComplex alpha,
    magmaDoubleComplex *A, int lda,
    magmaDoubleComplex *x, int incx,
    magmaDoubleComplex beta,
    magmaDoubleComplex *y, int incy,
    magmaDoubleComplex *WC,
    int my_gpu_id,
    int num_gpus,
    int nb,
    int the_chosen_block_id,
    int the_chosen_gpu_id,
    int offset)
{
#if (__CUDA_ARCH__ >= 200)
    /*
    if ( blockIdx.x < the_chosen_block_id ) {
        return;
    }
    */
    int i;
    int tx  = threadIdx.x;
    int ind = blockIdx.x * thread_x + tx;
    magmaDoubleComplex Ca;

    Ca = MAGMA_Z_ZERO;
    WC += ind + lda * blockIdx.x;
    
    for(i = blockIdx.x*thread_x; i < n; i += thread_x) {
        Ca += WC[0];
        WC += thread_x;
    }
    
    if ( ind < n  && ind >= offset )
        y[ind * incy] = beta * y[ind * incy]  + alpha * Ca;
#endif /* (__CUDA_ARCH__ >= 200) */
}


extern "C"
void magmablas_zhemv_fermi_L_mgpu_offset(
    magma_int_t n, magmaDoubleComplex alpha,
    magmaDoubleComplex *A, magma_int_t lda,
    magmaDoubleComplex *x, magma_int_t incx,
    magmaDoubleComplex beta,
    magmaDoubleComplex *y, magma_int_t incy,
    magmaDoubleComplex *dwork,
    magma_int_t my_gpu_id,
    magma_int_t num_gpus,
    magma_int_t nb,
    magma_int_t offset,
    magma_int_t num_blocks_skipped )
{
    magma_int_t the_chosen_block_id = offset / 64;
    magma_int_t the_chosen_gpu_id = the_chosen_block_id % num_gpus;
    magma_int_t kstan = offset % 64;

    /*
    printf("Enter magmablas_zhemv_fermi_L_mgpu_offset\n");
    printf("the_chosen_block_id = %d\n", the_chosen_block_id);
    printf("the_chosen_gpu_id = %d\n", the_chosen_gpu_id);
    printf("kstan = %d\n", kstan);
    */

    A += lda * num_blocks_skipped * 64 + the_chosen_block_id * 64;
    x += the_chosen_block_id * 64;
    y += the_chosen_block_id * 64;

    magma_int_t blocks = (n - 1)/hemv_bs + 1;
    blocks -= the_chosen_block_id;

    dim3 grid(blocks, 1, 1);
    dim3 threads(thread_x, thread_y, 1);
    dim3 threads_u(hemv_bs, 1, 1);

    the_chosen_block_id = 0;
    the_chosen_gpu_id = 0;

    /*
     * If matrix size is multiple of hemv_bs, we use a specific code.
     * otherwise, we call the generic case.
     */
    if ( n % hemv_bs == 0 ) {
        zhemv_kernel_fermi_L_special_mgpu_offset<<< grid, threads, 0, magma_stream >>>(
            n, alpha, A, lda, x, incx, beta, y, incy, dwork,
            my_gpu_id, num_gpus, nb,
            the_chosen_block_id, the_chosen_gpu_id, kstan);
    }
    else {
        magma_int_t m_mod_thread_x = (n % hemv_bs) - 1;
        zhemv_kernel_fermi_L_generic_mgpu_offset<<< grid, threads, 0, magma_stream >>>(
            n, alpha, A, lda, x, incx, beta, y, incy, dwork,
            m_mod_thread_x, my_gpu_id, num_gpus, nb,
            the_chosen_block_id, the_chosen_gpu_id, kstan);
    }

    zhemv_kernel_fermi_L_update_mgpu_offset<<< grid, threads_u, 0, magma_stream >>>(
        n, alpha, A, lda, x, incx, beta, y, incy, dwork,
        my_gpu_id, num_gpus, nb,
        the_chosen_block_id, the_chosen_gpu_id, kstan);
}


/**
    Purpose
    -------
    magmablas_zhemv  performs the matrix-vector operation:

        y := alpha*A*x + beta*y,

    where alpha and beta are scalars, x and y are n element vectors and
    A is an n by n hermitian matrix.

    Arguments
    ----------
    @param[in]
    uplo    magma_uplo_t.
            On entry, UPLO specifies whether the upper or lower
            triangular part of the array A is to be referenced as
            follows:
      -     = MagmaUpper:  Only the upper triangular part of A is to be referenced.
      -     = MagmaLower:  Only the lower triangular part of A is to be referenced.

    @param[in]
    n       INTEGER.
            On entry, N specifies the order of the matrix A.
            N must be at least zero.

    @param[in]
    alpha   COMPLEX*16.
            On entry, ALPHA specifies the scalar alpha.

    @param[in]
    A       COMPLEX*16 array of DIMENSION ( LDA, n ).
            Before entry with  UPLO = MagmaUpper, the leading n by n
            upper triangular part of the array A must contain the upper
            triangular part of the hermitian matrix and the strictly
            lower triangular part of A is not referenced.
            Before entry with UPLO = MagmaLower, the leading n by n
            lower triangular part of the array A must contain the lower
            triangular part of the hermitian matrix and the strictly
            upper triangular part of A is not referenced.
            Note that the imaginary parts of the diagonal elements need
            not be set and are assumed to be zero.

    @param[in]
    lda     INTEGER.
            On entry, LDA specifies the first dimension of A as declared
            in the calling (sub) program. LDA must be at least
            max( 1, n ).
            It is recommended that lda is multiple of 16. Otherwise
            performance would be deteriorated as the memory accesses
            would not be fully coalescent.

    @param[in]
    x       COMPLEX*16 array of dimension at least
            ( 1 + ( n - 1 )*abs( INCX ) ).
            Before entry, the incremented array X must contain the n
            element vector x.

    @param[in]
    incx    INTEGER.
            On entry, INCX specifies the increment for the elements of
            X. INCX must not be zero.

    @param[in]
    beta    COMPLEX*16.
            On entry, BETA specifies the scalar beta. When BETA is
            supplied as zero then Y need not be set on input.

    @param[in,out]
    y       COMPLEX*16 array of dimension at least
            ( 1 + ( n - 1 )*abs( INCY ) ).
            Before entry, the incremented array Y must contain the n
            element vector y. On exit, Y is overwritten by the updated
            vector y.

    @param[in]
    incy    INTEGER.
            On entry, INCY specifies the increment for the elements of
            Y. INCY must not be zero.

    @ingroup magma_zblas2
    ********************************************************************/
extern "C"
magma_int_t
magmablas_zhemv_mgpu_offset(
    magma_uplo_t uplo, magma_int_t n,
    magmaDoubleComplex alpha,
    magmaDoubleComplex **A, magma_int_t lda,
    magmaDoubleComplex **x, magma_int_t incx,
    magmaDoubleComplex beta,
    magmaDoubleComplex **y, magma_int_t incy,
    magmaDoubleComplex **work, magma_int_t lwork,
    magma_int_t num_gpus,
    magma_int_t nb,
    magma_int_t offset,
    magma_queue_t stream[][10])
{
    magma_int_t arch = magma_getdevice_arch();
    if ( arch < 200  ) {
        // --------------------
        // no CUDA ARCH 1.x version
        fprintf( stderr, "%s not supported on CUDA arch 1.x\n", __func__ );
        return MAGMA_ERR_NOT_SUPPORTED;
    }
        
    // --------------------
    // CUDA ARCH 2.x (Fermi) version
    int upper = (uplo == MagmaUpper);

    /*
     * Test the input parameters.
     */
    if ( (! upper) && (uplo != MagmaLower) ) {
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
    if ( (n == 0) || ( MAGMA_Z_EQUAL(alpha, MAGMA_Z_ZERO) && MAGMA_Z_EQUAL(beta, MAGMA_Z_ONE) ) )
        return MAGMA_SUCCESS;

    /* TODO: Upper case is not implemented in MAGMA */
    if ( upper ) {
        fprintf( stderr, "Upper case is not implemented on multi GPUs\n" );
        return MAGMA_ERR_NOT_IMPLEMENTED;
    }
    else {
        magma_int_t blocks = (n - 1)/thread_x + 1;
        magma_int_t lwmin  = lda * (blocks + 1);

        if ( lwork < lwmin ) {
            fprintf( stderr, "Not enough work space in %s: passed %d, required %d\n",
                     __func__, (int) lwork, (int) lwmin);
            return -12;
        }
        if ( nb != 64 ) {
            fprintf( stderr, "Error in %s: nb != 64, please reallocate matrix among GPUs\n", __func__ );
            return MAGMA_ERR_ILLEGAL_VALUE;
        }

        {
            magma_int_t i = 0;
            for(i=0; i < num_gpus; i++) {
                magma_setdevice(i);
                magmablasSetKernelStream(stream[i][0]);
    
                magma_int_t the_chosen_block_id = offset / 64;
                magma_int_t the_chosen_gpu_id = the_chosen_block_id % num_gpus;
    
                magma_int_t  num_blocks_skipped = the_chosen_block_id / num_gpus;
    
                if ( i < the_chosen_gpu_id ) {
                    num_blocks_skipped += 1;
                }
                
                int new_gpu_id = ( i + num_gpus - the_chosen_gpu_id ) % num_gpus;
                
                magmablas_zhemv_fermi_L_mgpu_offset(n, alpha, A[i], lda, x[i], incx, beta, y[i], incy, work[i],
                    new_gpu_id, num_gpus, nb, offset, num_blocks_skipped);
            }
        }
    }
    return MAGMA_SUCCESS;
}

extern "C"
magma_int_t
magmablas_zhemv2_mgpu_offset(
    magma_uplo_t uplo, magma_int_t n,
    magmaDoubleComplex alpha,
    magmaDoubleComplex **A, magma_int_t lda,
    magmaDoubleComplex **x, magma_int_t incx,
    magmaDoubleComplex beta,
    magmaDoubleComplex **y, magma_int_t incy,
    magmaDoubleComplex **work, magma_int_t lwork,
    magma_int_t num_gpus,
    magma_int_t nb,
    magma_int_t offset)
{
    magma_int_t arch = magma_getdevice_arch();
    if ( arch < 200  ) {
        // --------------------
        // no CUDA ARCH 1.x version
        fprintf( stderr, "%s not supported on CUDA arch 1.x\n", __func__ );
        return MAGMA_ERR_NOT_SUPPORTED;
    }
        
    // --------------------
    // CUDA ARCH 2.x (Fermi) version
    int upper = (uplo == MagmaUpper);

    /*
     * Test the input parameters.
     */
    if ( (! upper) && (uplo != MagmaLower) ) {
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
    if ( (n == 0) || ( MAGMA_Z_EQUAL(alpha, MAGMA_Z_ZERO) && MAGMA_Z_EQUAL(beta, MAGMA_Z_ONE) ) )
        return MAGMA_SUCCESS;

    /* TODO: Upper case is not implemented in MAGMA */
    if ( upper ) {
        fprintf( stderr, "Upper case is not implemented on multi GPUs\n" );
        return MAGMA_ERR_NOT_IMPLEMENTED;
    }
    else {
        magma_int_t blocks = (n - 1)/thread_x + 1;
        magma_int_t lwmin  = lda * (blocks + 1);

        if ( lwork < lwmin ) {
            fprintf( stderr, "Not enough work space in %s: passed %d, required %d\n",
                     __func__, (int) lwork, (int) lwmin);
            return -12;
        }
        if ( nb != 64 ) {
            fprintf( stderr, "Error in %s: nb != 64, please reallocate matrix among GPUs\n", __func__ );
            return MAGMA_ERR_ILLEGAL_VALUE;
        }
        
        if ( num_gpus == 1 ) {
            magmablas_zhemv_work(uplo, n-offset, alpha, A[0] + offset + lda * offset, lda, x[0] + offset, incx, beta, y[0] + offset, incy, work[0], lwork);
        }
        else {
            magma_int_t i = 0;
            for(i=0; i < num_gpus; i++) {
                magma_setdevice(i);
    
                magma_int_t the_chosen_block_id = offset / 64;
                magma_int_t the_chosen_gpu_id = the_chosen_block_id % num_gpus;
                
                magma_int_t  num_blocks_skipped = the_chosen_block_id / num_gpus;
                
                if ( i < the_chosen_gpu_id ) {
                    num_blocks_skipped += 1;
                }
                
                int new_gpu_id = ( i + num_gpus - the_chosen_gpu_id ) % num_gpus;
                
                magmablas_zhemv_fermi_L_mgpu_offset(n, alpha, A[i], lda, x[i], incx, beta, y[i], incy, work[i],
                    new_gpu_id, num_gpus, nb, offset, num_blocks_skipped);
            }
        }
    }
    return MAGMA_SUCCESS;
}


extern "C"
magma_int_t
magmablas_zhemv2_mgpu(
    magma_uplo_t uplo, magma_int_t n,
    magmaDoubleComplex alpha,
    magmaDoubleComplex **A, magma_int_t lda,
    magmaDoubleComplex **x, magma_int_t incx,
    magmaDoubleComplex beta,
    magmaDoubleComplex **y, magma_int_t incy,
    magmaDoubleComplex **work, magma_int_t lwork,
    magma_int_t num_gpus,
    magma_int_t nb)
{
    magma_int_t arch = magma_getdevice_arch();
    if ( arch < 200  ) {
        // --------------------
        // no CUDA ARCH 1.x version
        fprintf( stderr, "%s not supported on CUDA arch 1.x\n", __func__ );
        return MAGMA_ERR_NOT_SUPPORTED;
    }
    
    // --------------------
    // CUDA ARCH 2.x (Fermi) version
    int upper = (uplo == MagmaUpper);

    /*
     * Test the input parameters.
     */
    if ( (! upper) && (uplo != MagmaLower) ) {
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
    if ( (n == 0) || ( MAGMA_Z_EQUAL(alpha, MAGMA_Z_ZERO) && MAGMA_Z_EQUAL(beta, MAGMA_Z_ONE) ) )
        return MAGMA_SUCCESS;

    /* TODO: Upper case is not implemented in MAGMA */
    if ( upper ) {
        fprintf( stderr, "Upper case is not implemented on multi GPUs\n" );
        return MAGMA_ERR_NOT_IMPLEMENTED;
    }
    else {
        magma_int_t blocks = (n - 1)/thread_x + 1;
        magma_int_t lwmin  = lda * (blocks + 1);

        if ( lwork < lwmin ) {
            fprintf( stderr, "Not enough work space in %s: passed %d, required %d\n",
                     __func__, (int) lwork, (int) lwmin);
            return -12;
        }
        if ( nb != 64 ) {
            fprintf( stderr, "Error in %s: nb != 64, please reallocate matrix among GPUs\n", __func__ );
            return MAGMA_ERR_ILLEGAL_VALUE;
        }

        magma_int_t i = 0;
        for(i=0; i < num_gpus; i++) {
            magma_setdevice(i);

            magmablas_zhemv_fermi_L_mgpu_offset(n, alpha, A[i], lda, x[i], incx, beta, y[i], incy, work[i],
                i, num_gpus, nb, 0, 0);
        }
    }
    
    return MAGMA_SUCCESS;
}
