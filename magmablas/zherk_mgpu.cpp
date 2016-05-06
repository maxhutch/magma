/*
    -- MAGMA (version 2.0.2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date May 2016
       
       @author Azzam Haidar
       @author Ichi Yamazaki

       @precisions normal z -> s d c

*/
#include "magma_internal.h"
#include "trace.h"

/**
    Purpose
    -------
    This zherk_mgpu is internal routine used by zpotrf_mgpu_right.
    It has specific assumption on the block diagonal.
    
    @ingroup magma_zblas3_internal
    ********************************************************************/
extern "C" void
magma_zherk_mgpu(
    magma_int_t ngpu,
    magma_uplo_t uplo, magma_trans_t trans, magma_int_t nb, magma_int_t n, magma_int_t k,
    double alpha,
    magmaDoubleComplex_ptr dB[], magma_int_t lddb, magma_int_t b_offset,
    double beta,
    magmaDoubleComplex_ptr dC[], magma_int_t lddc, magma_int_t c_offset,
    magma_int_t nqueue, magma_queue_t queues[][10])
{
#define dB(id, i, j)  (dB[(id)]+(j)*lddb + (i)+b_offset)
#define dC(id, i, j)  (dC[(id)]+(j)*lddc + (i))
#define STREAM_ID(i) (nqueue > 1 ? 1+((i)/nb)%(nqueue-1) : 0)

    magma_int_t i, id, ib, ii, kk, n1;
    magmaDoubleComplex z_alpha = MAGMA_Z_MAKE(alpha,0.0);
    magmaDoubleComplex z_beta  = MAGMA_Z_MAKE(beta, 0.0);

    magma_device_t orig_dev;
    magma_getdevice( &orig_dev );
    
    /* diagonal update */
    for( i=0; i < n; i += nb ) {
        id = ((i+c_offset)/nb)%ngpu;
        kk = STREAM_ID( i+c_offset );

        ib = min(nb, n-i);
        ii = nb*((i+c_offset)/(nb*ngpu));

        /* zher2k on diagonal block */
        magma_setdevice(id);
        trace_gpu_start( id, kk, "syr2k", "syr2k" );
        magma_zherk( uplo, trans, ib, k,
                     alpha,  dB(id, i,          0 ), lddb,
                      beta,  dC(id, i+c_offset, ii), lddc, queues[id][kk] );
        trace_gpu_end( id, kk );
    }

    /* off-diagonal update */
    if (uplo == MagmaUpper) {
        for( i=nb; i < n; i += nb ) {
            id = ((i+c_offset)/nb)%ngpu;
            kk = STREAM_ID( i+c_offset );

            ib = min(nb, n-i);
            ii = nb*((i+c_offset)/(nb*ngpu));

            magma_setdevice(id);
            magma_zgemm( MagmaNoTrans, MagmaConjTrans, i, ib, k,
                         z_alpha, dB(id, 0, 0 ), lddb,
                                  dB(id, i, 0 ), lddb,
                         z_beta,  dC(id, 0, ii), lddc, queues[id][kk] );
        }
    }
    else {
        for( i=0; i < n-nb; i += nb ) {
            id = ((i+c_offset)/nb)%ngpu;
            kk = STREAM_ID( i+c_offset );

            ib = min(nb, n-i);
            ii = nb*((i+c_offset)/(nb*ngpu));
            n1 = n-i-ib;

            /* zgemm on off-diagonal blocks */
            magma_setdevice(id);
            trace_gpu_start( id, kk, "gemm_up", "gemm_up" );
            magma_zgemm( MagmaNoTrans, MagmaConjTrans, n1, ib, k,
                         z_alpha, dB(id, i+ib,           0 ), lddb,
                                  dB(id,  i,             0 ), lddb,
                         z_beta,  dC(id,  i+c_offset+ib, ii), lddc, queues[id][kk] );
            trace_gpu_end( id, kk );
        }
    }

    // TODO why not sync?
    //for( id=0; id < ngpu; id++ ) {
    //    magma_setdevice(id);
    //    //for( kk=0; kk < nqueue; kk++ )
    //    //    magma_queue_sync( queues[id][kk] );
    //}
    magma_setdevice( orig_dev );
}
#undef dB
#undef dC
#undef STREAM_ID


// ----------------------------------------------------------------------
extern "C" void
magma_zherk_mgpu2(
    magma_int_t ngpu,
    magma_uplo_t uplo, magma_trans_t trans, magma_int_t nb, magma_int_t n, magma_int_t k,
    double alpha,
    magmaDoubleComplex_ptr dB[], magma_int_t lddb, magma_int_t b_offset,
    double beta,
    magmaDoubleComplex_ptr dC[], magma_int_t lddc, magma_int_t c_offset,
    magma_int_t nqueue, magma_queue_t queues[][10])
{
#define dB(id, i, j)  (dB[(id)]+(j)*lddb + (i)+b_offset)
#define dC(id, i, j)  (dC[(id)]+(j)*lddc + (i))
#define STREAM_ID(i) (nqueue > 1 ? 1+((i)/nb)%(nqueue-1) : 0)

    magma_int_t i, id, ib, ii, kk, n1;
    magmaDoubleComplex z_alpha = MAGMA_Z_MAKE(alpha,0.0);
    magmaDoubleComplex z_beta  = MAGMA_Z_MAKE(beta, 0.0);

    magma_device_t orig_dev;
    magma_getdevice( &orig_dev );
    
    /* diagonal update */
    for( i=0; i < n; i += nb ) {
        id = ((i+c_offset)/nb)%ngpu;
        kk = STREAM_ID( i+c_offset );

        ib = min(nb, n-i);
        ii = nb*((i+c_offset)/(nb*ngpu));
    }

    if (uplo == MagmaUpper) {
        for( i=0; i < n; i += nb ) {
            id = ((i+c_offset)/nb)%ngpu;
            kk = STREAM_ID( i+c_offset );

            ib = min(nb, n-i);
            ii = nb*((i+c_offset)/(nb*ngpu));
            n1 = i+ib;

            magma_setdevice(id);

            /* zgemm on diag and off-diagonal blocks */
            magma_zgemm( MagmaNoTrans, MagmaConjTrans, n1, ib, k,
                         z_alpha, dB(id, 0, 0 ), lddb,
                                  dB(id, i, 0 ), lddb,
                         z_beta,  dC(id, 0, ii), lddc, queues[id][kk] );
        }
    }
    else {
        for( i=0; i < n; i += nb ) {
            id = ((i+c_offset)/nb)%ngpu;
            kk = STREAM_ID( i+c_offset );

            ib = min(nb, n-i);
            ii = nb*((i+c_offset)/(nb*ngpu));
            n1 = n-i;

            magma_setdevice(id);
            
            trace_gpu_start( id, kk, "gemm_up", "gemm_up" );
            /* zgemm on diag and off-diagonal blocks */
            magma_zgemm( MagmaNoTrans, MagmaConjTrans, n1, ib, k,
                         z_alpha, dB(id, i,           0), lddb,
                                  dB(id, i,           0), lddb,
                         z_beta,  dC(id, i+c_offset, ii), lddc, queues[id][kk] );
            trace_gpu_end( id, kk );
        }
    }

    // TODO: why not sync?
    //for( id=0; id < ngpu; id++ ) {
    //    magma_setdevice(id);
    //    //for( kk=0; kk < nqueue; kk++ )
    //    //    magma_queue_sync( queues[id][kk] );
    //}
    magma_setdevice( orig_dev );
}

#undef dB
#undef dC
#undef STREAM_ID
