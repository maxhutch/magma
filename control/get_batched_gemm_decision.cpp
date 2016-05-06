/*
    -- MAGMA (version 2.0.2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date May 2016
       
       @author Azzam Haidar
       @author Ahmad Abdelfattah
       
*/

#include "magma_internal.h"

#ifdef __cplusplus
extern "C" {
#endif

// ==== Definition of blocking sizes for Nvidia cards

// Advisory functions used to determine if cuBLAS should be used for batched gemm
// Decision is based on the dimensions and the shape
// Two cuBLAS-based alternatives are used (batched vs streamed)
// Decisions are based on tuning experiments conducted on Kepler K40c (bunsen).

// helper function - intended for internal use only
magma_int_t magma_get_gemm_shape(magma_trans_t transA, magma_trans_t transB)
{
    magma_int_t shape = -1;
    if      (transA == MagmaNoTrans   && transB == MagmaNoTrans)   { shape = 0; } // nn
    else if (transA == MagmaNoTrans   && transB == MagmaTrans)     { shape = 1; } // nt
    else if (transA == MagmaNoTrans   && transB == MagmaConjTrans) { shape = 2; } // nc
    else if (transA == MagmaTrans     && transB == MagmaNoTrans)   { shape = 3; } // tn
    else if (transA == MagmaTrans     && transB == MagmaTrans)     { shape = 4; } // tt
    else if (transA == MagmaTrans     && transB == MagmaConjTrans) { shape = 5; } // tc
    else if (transA == MagmaConjTrans && transB == MagmaNoTrans)   { shape = 6; } // cn
    else if (transA == MagmaConjTrans && transB == MagmaTrans)     { shape = 7; } // ct
    else if (transA == MagmaConjTrans && transB == MagmaConjTrans) { shape = 8; } // cc
    
    return shape;
}
///////////////////////////////////////////////////////////////////////////////////////////////////
// magma_Xrecommend_cublas_gemm_batched decides which is better (magma or cublas_batched),
// regardless of the performance of cublas stream
///////////////////////////////////////////////////////////////////////////////////////////////////
magma_int_t magma_srecommend_cublas_gemm_batched(
            magma_trans_t transa, magma_trans_t transb,
            magma_int_t m, magma_int_t n, magma_int_t k)
{
    magma_int_t use_cublas_gemm_batched = 0;
    magma_int_t shape = magma_get_gemm_shape(transa, transb);
    
    switch(shape)
    {
        case 0: // nn
            {
                use_cublas_gemm_batched = (magma_int_t) ( ( !(  k == 8  &&  n == 24 ) ) // ! (k == 8, n == 24)
                                                       && ( !(  k == 32             ) ) // ! k == 32
                                                        );
            }
            break;
        
        case 1: // nt
        case 2: // nc
            {
                use_cublas_gemm_batched = (magma_int_t) ( ( !(  k == 8  &&  n == 24 ) ) // ! (k == 8, n == 24)
                                                       && ( !(  k == 32             ) ) // ! k == 32
                                                        );
            }
            break;
        
        case 3: // tn
        case 6: // cn
            {
                use_cublas_gemm_batched = (magma_int_t) ( ( !(  k == 8  &&  n == 24 ) ) // ! (k == 8, n == 24)
                                                       && (  (  k < 32 )              ) // k < 32
                                                        );
            }
            break;
            
        case 4: // tt
        case 5: // tc
        case 7: // ct
        case 8: // cc
            {
                use_cublas_gemm_batched = (magma_int_t) ( ( !(  k == 8  &&  n == 24 ) ) // ! (k == 8, n == 24)
                                                       && ( !(  k == 32             ) ) // ! k == 32
                                                        );
            }
            break;
        
        default:;
    }
    return use_cublas_gemm_batched;
}
///////////////////////////////////////////////////////////////////////////////////////////////////
magma_int_t magma_drecommend_cublas_gemm_batched(
            magma_trans_t transa, magma_trans_t transb,
            magma_int_t m, magma_int_t n, magma_int_t k)
{
    magma_int_t use_cublas_gemm_batched = 0;
    // cublas batched is not used anywhere for DP
    return use_cublas_gemm_batched;
}
///////////////////////////////////////////////////////////////////////////////////////////////////
magma_int_t magma_crecommend_cublas_gemm_batched(
            magma_trans_t transa, magma_trans_t transb,
            magma_int_t m, magma_int_t n, magma_int_t k)
{
    magma_int_t use_cublas_gemm_batched = 0;
    magma_int_t shape = magma_get_gemm_shape(transa, transb);
    
    switch(shape)
    {
        case 0: // nn
            {
                use_cublas_gemm_batched = (magma_int_t) ( ( !(  k == 8  &&  n == 24 ) ) // ! (k == 8, n == 24)
                                                       && (  (  k < 32 )              ) // k < 32
                                                        );
            }
            break;
        
        case 1: // nt
        case 2: // nc
            {
                // No cublas batched for this case
                use_cublas_gemm_batched = 0;
            }
            break;
        
        case 3: // tn
        case 6: // cn
            {
                use_cublas_gemm_batched = (magma_int_t) ( ( k == 16 && n == 16 ) // k == 16, n == 16
                                                        );
            }
            break;
            
        case 4: // tt
        case 5: // tc
        case 7: // ct
        case 8: // cc
            {
                use_cublas_gemm_batched = (magma_int_t) ( ( !(  k == 8  &&  n == 24 ) ) // ! (k == 8, n == 24)
                                                       && (  (  k < 32 )              ) // k < 32
                                                        );
            }
            break;
        
        default:;
    }
    return use_cublas_gemm_batched;
}
///////////////////////////////////////////////////////////////////////////////////////////////////
magma_int_t magma_zrecommend_cublas_gemm_batched(
            magma_trans_t transa, magma_trans_t transb,
            magma_int_t m, magma_int_t n, magma_int_t k)
{
    magma_int_t use_cublas_gemm_batched = 0;
    magma_int_t shape = magma_get_gemm_shape(transa, transb);
    
    switch(shape)
    {
        case 0: // nn
            {
                use_cublas_gemm_batched = 1;
            }
            break;
        
        case 1: // nt
        case 2: // nc
            {
                use_cublas_gemm_batched = 1;
            }
            break;
        
        case 3: // tn
        case 6: // cn
            {
                use_cublas_gemm_batched = (magma_int_t)  (k < 32 ); // k < 32
            }
            break;
            
        case 4: // tt
        case 5: // tc
        case 7: // ct
        case 8: // cc
            {
                use_cublas_gemm_batched = 1;
            }
            break;
        
        default:;
    }
    return use_cublas_gemm_batched;
}
///////////////////////////////////////////////////////////////////////////////////////////////////
// magma_Xrecommend_cublas_gemm_stream decides if cublas stream should be used for a given
// gemm dimension/shape
///////////////////////////////////////////////////////////////////////////////////////////////////
magma_int_t magma_srecommend_cublas_gemm_stream(
            magma_trans_t transa, magma_trans_t transb,
            magma_int_t m, magma_int_t n, magma_int_t k)
{
    magma_int_t use_cublas_gemm_stream = 0;
    magma_int_t shape = magma_get_gemm_shape(transa, transb);
    
    switch(shape)
    {
        case 0: // nn
            {
                use_cublas_gemm_stream = (magma_int_t) ( ( (k >  32  && k <= 64 ) && (m >= 224         ) ) // k == 64, m >= 224
                                                      || ( (k >= 128            ) && (m >= 160         ) ) // k >= 128
                                                        );
            }
            break;
        
        case 1: // nt
        case 2: // nc
            {
                use_cublas_gemm_stream = (magma_int_t) ( ( (k >  32  && k <= 64 ) && (m >=  224 && m < 512) ) // k == 64, m == 224:512
                                                      || ( (k >= 128            ) && (m >= 224            ) ) // k >= 128, m >= 224
                                                        );
            }
            break;
        
        case 3: // tn
        case 6: // cn
            {
                use_cublas_gemm_stream = (magma_int_t) ( ( (k >  32  && k <= 64 ) && (m >= 192 && m <  512) ) // k == 64, m == 192:512
                                                      || ( (k >= 128            ) && (m >= 128            ) ) // k >= 128, m >= 224
                                                        );
            }
            break;
            
        case 4: // tt
        case 5: // tc
        case 7: // ct
        case 8: // cc
            {
                use_cublas_gemm_stream = (magma_int_t) ( ( (k >  32  && k <= 64 ) && (m >= 192 && m <  512) ) // k == 64, m == 192:512
                                                      || ( (k >= 128            ) && (m >= 160            ) ) // k >= 128, m >= 224
                                                        );
            }
            break;
        
        default:;
    }
    return use_cublas_gemm_stream;
}
///////////////////////////////////////////////////////////////////////////////////////////////////
magma_int_t magma_drecommend_cublas_gemm_stream(
            magma_trans_t transa, magma_trans_t transb,
            magma_int_t m, magma_int_t n, magma_int_t k)
{
    magma_int_t use_cublas_gemm_stream = 0;
    magma_int_t shape = magma_get_gemm_shape(transa, transb);
    
    switch(shape)
    {
        case 0: // nn
            {
                use_cublas_gemm_stream = (magma_int_t) ( ( (k >  32  && k <= 64 ) && (m >= 192         ) ) // k == 64, m >= 192
                                                      || ( (k >= 128            ) && (m >= 128         ) ) // k >= 128, m >= 128
                                                        );
            }
            break;
        
        case 1: // nt
        case 2: // nc
            {
                use_cublas_gemm_stream = (magma_int_t) ( ( (k >  32  && k <= 64 ) && (m >= 160         ) ) // k == 64, m >= 160
                                                      || ( (k >= 128            ) && (m >= 128         ) ) // k >= 128, m >= 128
                                                        );
            }
            break;
        
        case 3: // tn
        case 6: // cn
            {
                use_cublas_gemm_stream = (magma_int_t) ( ( (k >  16  && k <= 32 ) && (m >= 256         ) ) // k == 32, m >= 256
                                                      || ( (k >  32  && k <= 64 ) && (m >= 192         ) ) // k == 64, m >= 192
                                                      || ( (k >= 128            ) && (m >= 128         ) ) // k >= 128, m >= 128
                                                        );
            }
            break;
            
        case 4: // tt
        case 5: // tc
        case 7: // ct
        case 8: // cc
            {
                use_cublas_gemm_stream = (magma_int_t) ( ( (k >  16  && k <= 32 ) && (m >= 192         ) ) // k == 32, m >= 192
                                                      || ( (k >  32  && k <= 64 ) && (m >= 160         ) ) // k == 64, m >= 160
                                                      || ( (k >= 128            ) && (m >= 128         ) ) // k >= 128, m >= 128
                                                        );
            }
            break;
        
        default:;
    }
    return use_cublas_gemm_stream;
}
///////////////////////////////////////////////////////////////////////////////////////////////////
magma_int_t magma_crecommend_cublas_gemm_stream(
            magma_trans_t transa, magma_trans_t transb,
            magma_int_t m, magma_int_t n, magma_int_t k)
{
    magma_int_t use_cublas_gemm_stream = 0;
    magma_int_t shape = magma_get_gemm_shape(transa, transb);
    
    switch(shape)
    {
        case 0: // nn
            {
                use_cublas_gemm_stream = (magma_int_t) ( ( (k >  16  && k <= 32 ) && (m >= 192         ) ) // k == 32, m >= 192
                                                      || ( (k >  32  && k <= 64 ) && (m >= 128         ) ) // k == 64, m >= 128
                                                      || ( (k >= 128            ) && (m >= 128         ) ) // k >= 128, m >= 128
                                                        );
            }
            break;
        
        case 1: // nt
        case 2: // nc
            {
                use_cublas_gemm_stream = (magma_int_t) ( ( (k >  16  && k <= 32 ) && (m >= 192         ) ) // k == 32, m >= 192
                                                      || ( (k >  32  && k <= 64 ) && (m >= 128         ) ) // k == 64, m >= 128
                                                      || ( (k >= 128            ) && (m >= 128         ) ) // k >= 128, m >= 128
                                                        );
            }
            break;
        
        case 3: // tn
        case 6: // cn
            {
                use_cublas_gemm_stream = (magma_int_t) ( ( (k >  32  && k <= 64 ) && (m >= 128         ) ) // k == 64, m >= 128
                                                      || ( (k >= 128            ) && (m >= 128         ) ) // k >= 128, m >= 128
                                                        );
            }
            break;
            
        case 4: // tt
        case 5: // tc
        case 7: // ct
        case 8: // cc
            {
                use_cublas_gemm_stream = (magma_int_t) ( ( (k >  16  && k <= 32 ) && (m >= 192         ) ) // k == 32, m >= 192
                                                      || ( (k >  32  && k <= 64 ) && (m >= 128         ) ) // k == 64, m >= 128
                                                      || ( (k >= 128            ) && (m >= 128         ) ) // k >= 128, m >= 128
                                                        );
            }
            break;
        
        default:;
    }
    return use_cublas_gemm_stream;
}
///////////////////////////////////////////////////////////////////////////////////////////////////
magma_int_t magma_zrecommend_cublas_gemm_stream(
            magma_trans_t transa, magma_trans_t transb,
            magma_int_t m, magma_int_t n, magma_int_t k)
{
    magma_int_t use_cublas_gemm_stream = 0;
    magma_int_t shape = magma_get_gemm_shape(transa, transb);
    
    switch(shape)
    {
        case 0: // nn
            {
                use_cublas_gemm_stream = (magma_int_t) ( ( (k >  16  && k <= 32 ) && (m >= 160         ) ) // k == 32, m >= 160
                                                      || ( (k >  32  && k <= 64 ) && (m >= 128         ) ) // k == 64, m >= 128
                                                      || ( (k >= 128            ) && (m >= 64         ) ) // k >= 128, m >= 64
                                                        );
            }
            break;
        
        case 1: // nt
        case 2: // nc
            {
                use_cublas_gemm_stream = (magma_int_t) ( ( (k >  16  && k <= 32 ) && (m >= 128         ) ) // k == 32, m >= 128
                                                      || ( (k >  32  && k <= 64 ) && (m >= 128         ) ) // k == 64, m >= 128
                                                      || ( (k >= 128            ) && (m >= 128         ) ) // k >= 128, m >= 128
                                                        );
            }
            break;
        
        case 3: // tn
        case 6: // cn
            {
                use_cublas_gemm_stream = (magma_int_t) ( ( (k >  16  && k <= 32 ) && (m >= 128         ) ) // k == 32, m >= 128
                                                      || ( (k >  32  && k <= 64 ) && (m >= 128         ) ) // k == 64, m >= 128
                                                      || ( (k >= 128            ) && (m >= 128         ) ) // k >= 128, m >= 128
                                                        );
            }
            break;
            
        case 4: // tt
        case 5: // tc
        case 7: // ct
        case 8: // cc
            {
                use_cublas_gemm_stream = (magma_int_t) ( ( (k >  16  && k <= 32 ) && (m >= 128         ) ) // k == 32, m >= 128
                                                      || ( (k >  32  && k <= 64 ) && (m >= 128         ) ) // k == 64, m >= 128
                                                      || ( (k >= 128            ) && (m >= 128         ) ) // k >= 128, m >= 128
                                                        );
            }
            break;
        
        default:;
    }
    return use_cublas_gemm_stream;
}
///////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef __cplusplus
} // extern "C"
#endif
