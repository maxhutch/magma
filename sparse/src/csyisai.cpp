/*
    -- MAGMA (version 2.2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date November 2016

       @author Hartwig Anzt

       @generated from sparse/src/zsyisai.cpp, normal z -> c, Sun Nov 20 20:20:46 2016
*/

#include "magmasparse_internal.h"

#include <cuda.h>  // for CUDA_VERSION

#define PRECISION_c


/***************************************************************************//**
    Purpose
    -------

    Prepares Incomplete Cholesky preconditioner using a sparse approximate 
    inverse instead of sparse triangular solves. This is the symmetric variant 
    of cgeisai.cpp. 
    

    Arguments
    ---------

    @param[in]
    A           magma_c_matrix
                input matrix A
                
    @param[in]
    b           magma_c_matrix
                input RHS b

    @param[in,out]
    precond     magma_c_preconditioner*
                preconditioner parameters
                
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_cgepr
    ********************************************************************/
extern "C"
magma_int_t
magma_cicisaisetup(
    magma_c_matrix A,
    magma_c_matrix b,
    magma_c_preconditioner *precond,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    // real_Double_t start, end;
    
    magmaFloatComplex *trisystems_d = NULL; 
    magmaFloatComplex *rhs_d = NULL; 
    magma_index_t *sizes_d = NULL, *locations_d = NULL; 
    magma_index_t *sizes_h = NULL; 
    magma_int_t maxsize, nnzloc, nnzL=0, nnzU=0;
    int warpsize=32;
    int offset = 0; // can be changed to better match the matrix structure
    magma_c_matrix LT={Magma_CSR}, MT={Magma_CSR}, QT={Magma_CSR};
    magma_int_t z;
    // magma_int_t timing = 1;
    
    
#if (CUDA_VERSION <= 6000) // this won't work, just to have something...
    printf( "%% error: ISAI preconditioner requires CUDA > 6.0.\n" );
    info = MAGMA_ERR_NOT_SUPPORTED;
    goto cleanup;
#endif

    // CHECK( magma_index_malloc( &sizes_d, A.num_rows ) );
    CHECK( magma_index_malloc_cpu( &sizes_h, A.num_rows+1 ) );
    // CHECK( magma_index_malloc( &locations_d, A.num_rows*warpsize ) );
    // CHECK( magma_cmalloc( &trisystems_d, min(320000,A.num_rows) *warpsize*warpsize ) ); // fixed size - go recursive
    // CHECK( magma_cmalloc( &rhs_d, A.num_rows*warpsize ) );
    
    for( magma_int_t i=0; i<A.num_rows; i++ ){
            maxsize = sizes_h[i] = 0;
    }
    
    // we need this in any case
    CHECK( magma_cmtranspose( precond->L, &MT, queue ) );
    
    // SPAI for L 
    if( precond->trisolver == Magma_JACOBI ){ // block diagonal structure
        if( precond->pattern == 0 ){
            precond->pattern = 1;    
        }
        // magma_cmisai_blockstruct_gpu( A.num_rows, -precond->pattern, offset, MagmaLower, &QT, queue );
        // magma_c_mvisu(QT, queue );
        // printf("done here\n");
        magma_cmisai_blockstruct( A.num_rows, precond->pattern, offset, MagmaLower, &MT, queue );
        CHECK( magma_c_mtransfer( MT, &QT, Magma_CPU, Magma_DEV, queue ) );
        magma_cmfree( &MT, queue );
        CHECK( magma_cmtranspose( QT, &MT, queue ) );
        magma_cmfree( &QT, queue );
    }
    else if (precond->trisolver == Magma_VBJACOBI) { // block diagonal structure with variable blocksize
        CHECK( magma_c_mtransfer( A, &QT, A.memory_location, Magma_CPU, queue ) );
        magma_cmfree( &MT, queue );
        CHECK( magma_cmsupernodal( &precond->pattern, QT, &MT, queue ) );
        magma_cmfree( &QT, queue );
        CHECK( magma_cmconvert( MT, &QT, Magma_CSR, Magma_CSRL, queue ) );
        magma_cmfree( &MT, queue );
        CHECK( magma_cmconvert( QT, &MT, Magma_CSR, Magma_CSR, queue ) );
        magma_cmfree( &QT, queue );
        CHECK( magma_c_mtransfer( MT, &QT, Magma_CPU, Magma_DEV, queue ) );
        magma_cmfree( &MT, queue );
        CHECK( magma_cmtranspose( QT, &MT, queue ) );
        magma_cmfree( &QT, queue );
    }
    else if (precond->trisolver == Magma_ISAI) {
        if( precond->pattern == 100 ){
            CHECK( magma_cgeisai_maxblock( LT, &MT, queue ) );
        }
        else {
            // pattern L^x
            // CHECK( magma_c_mtransfer( LT, &MT, Magma_DEV, Magma_DEV, queue ) );
            if( precond->pattern > 1 ){
                CHECK( magma_c_mtransfer( MT, &LT, Magma_DEV, Magma_DEV, queue ) );
                z = 1;
                while( z<precond->pattern ){
                    CHECK( magma_c_spmm( MAGMA_C_ONE, LT, MT, &QT, queue ) );
                    magma_cmfree( &MT, queue );
                    CHECK( magma_c_mtransfer( QT, &MT, Magma_DEV, Magma_DEV, queue ) );
                    magma_cmfree( &QT, queue );
                    z++;
                }
            }
        }
    }
    magma_index_getvector( A.num_rows+1, MT.drow, 1, sizes_h, 1, queue );
    maxsize = 0;
    for( magma_int_t i=0; i<A.num_rows; i++ ){
        nnzloc = sizes_h[i+1]-sizes_h[i];
        nnzL+= nnzloc;
        if( nnzloc > maxsize ){
            maxsize = sizes_h[i+1]-sizes_h[i];
        }
        if( maxsize > warpsize ){
            printf("%%   error for ISAI: size of system %d is too large by %d\n", (int) i, (int) (maxsize-32) ); 
            break;
        }
    }
    printf("%% nnz in L-ISAI: %d\t", (int) nnzL); 
    // this can be modified to the thread-block-size
    if( maxsize > warpsize ){
       info = -(maxsize - warpsize);     
       goto cleanup;
    }
 
    // via main memory
    //  if( maxsize <= 8 ){
    //      CHECK( magma_cisaigenerator_8_gpu( MagmaLower, MagmaNoTrans, MagmaNonUnit, 
    //                  LT, &MT, sizes_d, locations_d, trisystems_d, rhs_d, queue ) );
    //  } else if( maxsize <= 16 ){
    //      CHECK( magma_cisaigenerator_16_gpu( MagmaLower, MagmaNoTrans, MagmaNonUnit, 
    //                  LT, &MT, sizes_d, locations_d, trisystems_d, rhs_d, queue ) );
    //  } else {
    //      CHECK( magma_cisaigenerator_32_gpu( MagmaLower, MagmaNoTrans, MagmaNonUnit, 
    //                  LT, &MT, sizes_d, locations_d, trisystems_d, rhs_d, queue ) );
    //  }
    // via registers
    CHECK( magma_cisai_generator_regs( MagmaLower, MagmaNoTrans, MagmaNonUnit, 
                    precond->L, &MT, sizes_d, locations_d, trisystems_d, rhs_d, queue ) );
      
    CHECK( magma_cmtranspose( MT, &precond->LD, queue ) );
    magma_cmfree( &LT, queue );
    magma_cmfree( &MT, queue );
    //  magma_c_mvisu(precond->LD, queue);
   
    // we need this in any case
    CHECK( magma_cmtranspose( precond->U, &MT, queue ) );
    
    // SPAI for U 
    if( precond->trisolver == Magma_JACOBI ){ // block diagonal structure
        if( precond->pattern == 0 ){
            precond->pattern = 1;    
        }
        magma_cmisai_blockstruct( A.num_rows, precond->pattern, offset, MagmaUpper, &MT, queue );
        CHECK( magma_c_mtransfer( MT, &QT, Magma_CPU, Magma_DEV, queue ) );
        magma_cmfree( &MT, queue );
        CHECK( magma_cmtranspose( QT, &MT, queue ) );
        magma_cmfree( &QT, queue );
    }
    else if (precond->trisolver == Magma_VBJACOBI) { // block diagonal structure with variable blocksize
        CHECK( magma_c_mtransfer( A, &QT, A.memory_location, Magma_CPU, queue ) );
        magma_cmfree( &MT, queue );
        CHECK( magma_cmsupernodal( &precond->pattern, QT, &MT, queue ) );
        magma_cmfree( &QT, queue );
        CHECK( magma_cmconvert( MT, &QT, Magma_CSR, Magma_CSRU, queue ) );
        magma_cmfree( &MT, queue );
        CHECK( magma_cmconvert( QT, &MT, Magma_CSR, Magma_CSR, queue ) );
        magma_cmfree( &QT, queue );
        CHECK( magma_c_mtransfer( MT, &QT, Magma_CPU, Magma_DEV, queue ) );
        magma_cmfree( &MT, queue );
        CHECK( magma_cmtranspose( QT, &MT, queue ) );
        magma_cmfree( &QT, queue );
    }
    else if (precond->trisolver == Magma_ISAI) {
        if( precond->pattern == 100 ){
            CHECK( magma_cgeisai_maxblock( LT, &MT, queue ) );
        }
        else {
            // pattern U^x
            // CHECK( magma_c_mtransfer( LT, &MT, Magma_DEV, Magma_DEV, queue ) );
            if( precond->pattern > 1 ){
                CHECK( magma_c_mtransfer( MT, &LT, Magma_DEV, Magma_DEV, queue ) );
                z = 1;
                while( z<precond->pattern ){
                    CHECK( magma_c_spmm( MAGMA_C_ONE, LT, MT, &QT, queue ) );
                    magma_cmfree( &MT, queue );
                    CHECK( magma_c_mtransfer( QT, &MT, Magma_DEV, Magma_DEV, queue ) );
                    magma_cmfree( &QT, queue );
                    z++;
                }
            }
        }
    }
    magma_index_getvector( A.num_rows+1, MT.drow, 1, sizes_h, 1, queue );
    maxsize = 0;
    for( magma_int_t i=0; i<A.num_rows; i++ ){
        nnzloc = sizes_h[i+1]-sizes_h[i];
        nnzU+= nnzloc;
        if( nnzloc > maxsize ){
            maxsize = sizes_h[i+1]-sizes_h[i];
        }
        if( maxsize > warpsize ){
            printf("%%   error for ISAI: size of system %d is too large by %d\n", (int) i, (int) (maxsize-32) ); 
            break;
        }
    }
    printf("%% nnz in U-ISAI: %d\t", (int) nnzU); 
    // this can be modified to the thread-block-size
    if( maxsize > warpsize ){
       info = -(maxsize - warpsize);     
       goto cleanup;
    }
    
    // via main memory
    //   if( maxsize <= 8 ){
    //       CHECK( magma_cisaigenerator_8_gpu( MagmaUpper, MagmaNoTrans, MagmaNonUnit, 
    //                   LT, &MT, sizes_d, locations_d, trisystems_d, rhs_d, queue ) );
    //   } else if( maxsize <= 16 ){
    //       CHECK( magma_cisaigenerator_16_gpu( MagmaUpper, MagmaNoTrans, MagmaNonUnit, 
    //                   LT, &MT, sizes_d, locations_d, trisystems_d, rhs_d, queue ) );
    //   } else {
    //       CHECK( magma_cisaigenerator_32_gpu( MagmaUpper, MagmaNoTrans, MagmaNonUnit, 
    //                   LT, &MT, sizes_d, locations_d, trisystems_d, rhs_d, queue ) );
    //   }
    // via registers
    CHECK( magma_cisai_generator_regs( MagmaUpper, MagmaNoTrans, MagmaNonUnit, 
                    precond->U, &MT, sizes_d, locations_d, trisystems_d, rhs_d, queue ) );

    CHECK( magma_cmtranspose( MT, &precond->UD, queue ) );
    // magma_c_mvisu( precond->UD, queue ); 
     
cleanup:
    magma_free( sizes_d );
    magma_free_cpu( sizes_h );
    magma_free( locations_d );
    magma_free( trisystems_d );
    magma_free( rhs_d );
    magma_cmfree( &LT, queue );
    magma_cmfree( &MT, queue );
    magma_cmfree( &QT, queue );
    
    return info;
}
    
