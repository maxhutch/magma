/*
    -- MAGMA (version 1.6.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date November 2014

       @generated from testing_zspmv.cpp normal z -> c, Sat Nov 15 19:54:24 2014
       @author Hartwig Anzt
*/

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime_api.h>
#include <cublas.h>
#include <cusparse_v2.h>
#include <cuda_profiler_api.h>

#ifdef MAGMA_WITH_MKL
    #include <mkl_spblas.h>

    #define PRECISION_c
    #if defined(PRECISION_z)
    #define MKL_ADDR(a) (MKL_Complex8*)(a)
    #elif defined(PRECISION_c)
    #define MKL_ADDR(a) (MKL_Complex8*)(a)
    #else
    #define MKL_ADDR(a) (a)
    #endif
#endif

// includes, project
#include "flops.h"
#include "magma.h"
#include "magmasparse.h"
#include "magma_lapack.h"
#include "testings.h"


/* ////////////////////////////////////////////////////////////////////////////
   -- testing sparse matrix vector product
*/
int main(  int argc, char** argv )
{
    TESTING_INIT();
    magma_queue_t queue;
    magma_queue_create( /*devices[ opts->device ],*/ &queue );

    magma_c_sparse_matrix hA, hA_SELLP, hA_ELL, dA, dA_SELLP, dA_ELL;
    hA_SELLP.blocksize = 8;
    hA_SELLP.alignment = 8;
    real_Double_t start, end, res;
    magma_int_t *pntre;

    magmaFloatComplex c_one  = MAGMA_C_MAKE(1.0, 0.0);
    magmaFloatComplex c_zero = MAGMA_C_MAKE(0.0, 0.0);
    
    magma_int_t i, j;
    for( i = 1; i < argc; ++i ) {
        if ( strcmp("--blocksize", argv[i]) == 0 ) {
            hA_SELLP.blocksize = atoi( argv[++i] );
        } else if ( strcmp("--alignment", argv[i]) == 0 ) {
            hA_SELLP.alignment = atoi( argv[++i] );
        } else
            break;
    }
    printf( "\n#    usage: ./run_cspmv"
        " [ --blocksize %d --alignment %d (for SELLP) ]"
        " matrices \n\n", (int) hA_SELLP.blocksize, (int) hA_SELLP.alignment );

    while(  i < argc ) {

        if ( strcmp("LAPLACE2D", argv[i]) == 0 && i+1 < argc ) {   // Laplace test
            i++;
            magma_int_t laplace_size = atoi( argv[i] );
            magma_cm_5stencil(  laplace_size, &hA, queue );
        } else {                        // file-matrix test
            magma_c_csr_mtx( &hA,  argv[i], queue );
        }

        printf( "\n# matrix info: %d-by-%d with %d nonzeros\n\n",
                            (int) hA.num_rows,(int) hA.num_cols,(int) hA.nnz );

        real_Double_t FLOPS = 2.0*hA.nnz/1e9;

        magma_c_vector hx, hy, dx, dy, hrefvec, hcheck;

        // init CPU vectors
        magma_c_vinit( &hx, Magma_CPU, hA.num_rows, c_zero, queue );
        magma_c_vinit( &hy, Magma_CPU, hA.num_rows, c_zero, queue );

        // init DEV vectors
        magma_c_vinit( &dx, Magma_DEV, hA.num_rows, c_one, queue );
        magma_c_vinit( &dy, Magma_DEV, hA.num_rows, c_zero, queue );

        #ifdef MAGMA_WITH_MKL
            // calling MKL with CSR
            pntre = (magma_int_t*)malloc( (hA.num_rows+1)*sizeof(magma_int_t) );
            pntre[0] = 0;
            for (j=0; j<hA.num_rows; j++ ) {
                pntre[j] = hA.row[j+1];
            }
             MKL_INT num_rows = hA.num_rows;
             MKL_INT num_cols = hA.num_cols;
             MKL_INT nnz = hA.nnz;

            MKL_INT *col;
            TESTING_MALLOC_CPU( col, MKL_INT, nnz );
            for( magma_int_t t=0; t < hA.nnz; ++t ) {
                col[ t ] = hA.col[ t ];
            }
            MKL_INT *row;
            TESTING_MALLOC_CPU( row, MKL_INT, num_rows );
            for( magma_int_t t=0; t < hA.num_rows; ++t ) {
                row[ t ] = hA.col[ t ];
            }
    
            start = magma_wtime();
            for (j=0; j<10; j++ ) {
                mkl_ccsrmv( "N", &num_rows, &num_cols, 
                            MKL_ADDR(&c_one), "GFNC", MKL_ADDR(hA.val), 
                            col, row, pntre, 
                                                    MKL_ADDR(hx.val), 
                            MKL_ADDR(&c_zero),        MKL_ADDR(hy.val) );
            }
            end = magma_wtime();
            printf( "\n > MKL  : %.2e seconds %.2e GFLOP/s    (CSR).\n",
                                            (end-start)/10, FLOPS*10/(end-start) );

            TESTING_FREE_CPU( row );
            TESTING_FREE_CPU( col );
            free(pntre);
        #endif // MAGMA_WITH_MKL

        // copy matrix to GPU
        magma_c_mtransfer( hA, &dA, Magma_CPU, Magma_DEV, queue );        
        // SpMV on GPU (CSR) -- this is the reference!
        start = magma_sync_wtime( queue );
        for (j=0; j<10; j++)
            magma_c_spmv( c_one, dA, dx, c_zero, dy, queue );
        end = magma_sync_wtime( queue );
        printf( " > MAGMA: %.2e seconds %.2e GFLOP/s    (standard CSR).\n",
                                        (end-start)/10, FLOPS*10/(end-start) );
        magma_c_mfree(&dA, queue );
        magma_c_vtransfer( dy, &hrefvec , Magma_DEV, Magma_CPU, queue );

        // convert to ELL and copy to GPU
        magma_c_mconvert(  hA, &hA_ELL, Magma_CSR, Magma_ELL, queue );
        magma_c_mtransfer( hA_ELL, &dA_ELL, Magma_CPU, Magma_DEV, queue );
        magma_c_mfree(&hA_ELL, queue );
        magma_c_vfree( &dy, queue );
        magma_c_vinit( &dy, Magma_DEV, hA.num_rows, c_zero, queue );
        // SpMV on GPU (ELL)
        start = magma_sync_wtime( queue );
        for (j=0; j<10; j++)
            magma_c_spmv( c_one, dA_ELL, dx, c_zero, dy, queue );
        end = magma_sync_wtime( queue );
        printf( " > MAGMA: %.2e seconds %.2e GFLOP/s    (standard ELL).\n",
                                        (end-start)/10, FLOPS*10/(end-start) );
        magma_c_mfree(&dA_ELL, queue );
        magma_c_vtransfer( dy, &hcheck , Magma_DEV, Magma_CPU, queue );
        res = 0.0;
        for(magma_int_t k=0; k<hA.num_rows; k++ )
            res=res + MAGMA_C_REAL(hcheck.val[k]) - MAGMA_C_REAL(hrefvec.val[k]);
        if ( res < .000001 )
            printf("# tester spmv ELL:  ok\n");
        else
            printf("# tester spmv ELL:  failed\n");
        magma_c_vfree( &hcheck, queue );

        // convert to SELLP and copy to GPU
        magma_c_mconvert(  hA, &hA_SELLP, Magma_CSR, Magma_SELLP, queue );
        magma_c_mtransfer( hA_SELLP, &dA_SELLP, Magma_CPU, Magma_DEV, queue );
        magma_c_mfree(&hA_SELLP, queue );
        magma_c_vfree( &dy, queue );
        magma_c_vinit( &dy, Magma_DEV, hA.num_rows, c_zero, queue );
        // SpMV on GPU (SELLP)
        start = magma_sync_wtime( queue );
        for (j=0; j<10; j++)
            magma_c_spmv( c_one, dA_SELLP, dx, c_zero, dy, queue );
        end = magma_sync_wtime( queue );
        printf( " > MAGMA: %.2e seconds %.2e GFLOP/s    (SELLP).\n",
                                        (end-start)/10, FLOPS*10/(end-start) );

        magma_c_vtransfer( dy, &hcheck , Magma_DEV, Magma_CPU, queue );
        res = 0.0;
        for(magma_int_t k=0; k<hA.num_rows; k++ )
            res=res + MAGMA_C_REAL(hcheck.val[k]) - MAGMA_C_REAL(hrefvec.val[k]);
        printf("# |x-y|_F = %8.2e\n", res);
        if ( res < .000001 )
            printf("# tester spmv SELL-P:  ok\n");
        else
            printf("# tester spmv SELL-P:  failed\n");
        magma_c_vfree( &hcheck, queue );

        magma_c_mfree(&dA_SELLP, queue );


        // SpMV on GPU (CUSPARSE - CSR)
        // CUSPARSE context //

        cusparseHandle_t cusparseHandle = 0;
        cusparseStatus_t cusparseStatus;
        cusparseStatus = cusparseCreate(&cusparseHandle);
        cusparseSetStream( cusparseHandle, queue );

        cusparseMatDescr_t descr = 0;
        cusparseStatus = cusparseCreateMatDescr(&descr);

        cusparseSetMatType(descr,CUSPARSE_MATRIX_TYPE_GENERAL);
        cusparseSetMatIndexBase(descr,CUSPARSE_INDEX_BASE_ZERO);
        magmaFloatComplex alpha = c_one;
        magmaFloatComplex beta = c_zero;
        magma_c_vfree( &dy, queue );
        magma_c_vinit( &dy, Magma_DEV, hA.num_rows, c_zero, queue );

        // copy matrix to GPU
        magma_c_mtransfer( hA, &dA, Magma_CPU, Magma_DEV, queue );

        start = magma_sync_wtime( queue );
        for (j=0; j<10; j++)
            cusparseStatus =
            cusparseCcsrmv(cusparseHandle,CUSPARSE_OPERATION_NON_TRANSPOSE, 
                        hA.num_rows, hA.num_cols, hA.nnz, &alpha, descr, 
                        dA.dval, dA.drow, dA.dcol, dx.dval, &beta, dy.dval);
        end = magma_sync_wtime( queue );
        if (cusparseStatus != 0)    printf("error in cuSPARSE CSR\n");
        printf( " > CUSPARSE: %.2e seconds %.2e GFLOP/s    (CSR).\n",
                                        (end-start)/10, FLOPS*10/(end-start) );
        cusparseMatDescr_t descrA;
        cusparseStatus = cusparseCreateMatDescr(&descrA);
         if (cusparseStatus != 0)    printf("error\n");
        cusparseHybMat_t hybA;
        cusparseStatus = cusparseCreateHybMat( &hybA );
         if (cusparseStatus != 0)    printf("error\n");

        magma_c_vtransfer( dy, &hcheck , Magma_DEV, Magma_CPU, queue );
        res = 0.0;
        for(magma_int_t k=0; k<hA.num_rows; k++ )
            res=res + MAGMA_C_REAL(hcheck.val[k]) - MAGMA_C_REAL(hrefvec.val[k]);
        printf("# |x-y|_F = %8.2e\n", res);
        if ( res < .000001 )
            printf("# tester spmv cuSPARSE CSR:  ok\n");
        else
            printf("# tester spmv cuSPARSE CSR:  failed\n");
        magma_c_vfree( &hcheck, queue );
        magma_c_vfree( &dy, queue );
        magma_c_vinit( &dy, Magma_DEV, hA.num_rows, c_zero, queue );
       
        cusparseCcsr2hyb(cusparseHandle,  hA.num_rows, hA.num_cols,
                        descrA, dA.dval, dA.drow, dA.dcol,
                        hybA, 0, CUSPARSE_HYB_PARTITION_AUTO);

        start = magma_sync_wtime( queue );
        for (j=0; j<10; j++)
            cusparseStatus =
            cusparseChybmv( cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, 
               &alpha, descrA, hybA,
               dx.dval, &beta, dy.dval);
        end = magma_sync_wtime( queue );
        if (cusparseStatus != 0)    printf("error in cuSPARSE HYB\n");
        printf( " > CUSPARSE: %.2e seconds %.2e GFLOP/s    (HYB).\n",
                                        (end-start)/10, FLOPS*10/(end-start) );

        magma_c_vtransfer( dy, &hcheck , Magma_DEV, Magma_CPU, queue );
        res = 0.0;
        for(magma_int_t k=0; k<hA.num_rows; k++ )
            res=res + MAGMA_C_REAL(hcheck.val[k]) - MAGMA_C_REAL(hrefvec.val[k]);
        printf("# |x-y|_F = %8.2e\n", res);
        if ( res < .000001 )
            printf("# tester spmv cuSPARSE HYB:  ok\n");
        else
            printf("# tester spmv cuSPARSE HYB:  failed\n");
        magma_c_vfree( &hcheck, queue );

        cusparseDestroyMatDescr( descrA );
        cusparseDestroyHybMat( hybA );
        cusparseDestroy( cusparseHandle );

        magma_c_mfree(&dA, queue );



        printf("\n\n");


        // free CPU memory
        magma_c_mfree(&hA, queue );
        magma_c_vfree(&hx, queue );
        magma_c_vfree(&hy, queue );
        magma_c_vfree(&hrefvec, queue );
        // free GPU memory
        magma_c_vfree(&dx, queue );
        magma_c_vfree(&dy, queue );

        i++;

    }
    
    magma_queue_destroy( queue );
    TESTING_FINALIZE();
    return 0;
}
