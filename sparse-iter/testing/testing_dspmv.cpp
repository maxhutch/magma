/*
    -- MAGMA (version 1.7.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date September 2015

       @generated from testing_zspmv.cpp normal z -> d, Fri Sep 11 18:29:47 2015
       @author Hartwig Anzt
*/

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#ifdef MAGMA_WITH_MKL
    #include <mkl_spblas.h>

    #define PRECISION_d
    #if defined(PRECISION_z)
    #define MKL_ADDR(a) (double*)(a)
    #elif defined(PRECISION_c)
    #define MKL_ADDR(a) (MKL_Complex8*)(a)
    #else
    #define MKL_ADDR(a) (a)
    #endif
#endif

// includes, project
#include "flops.h"
#include "magma.h"
#include "magma_lapack.h"
#include "testings.h"
#include "common_magmasparse.h"


/* ////////////////////////////////////////////////////////////////////////////
   -- testing sparse matrix vector product
*/
int main(  int argc, char** argv )
{
    magma_int_t info = 0;
    TESTING_INIT();
    magma_queue_t queue=NULL;
    magma_queue_create( &queue );

    magma_d_matrix hA={Magma_CSR}, hA_SELLP={Magma_CSR}, hA_ELL={Magma_CSR}, 
    dA={Magma_CSR}, dA_SELLP={Magma_CSR}, dA_ELL={Magma_CSR};
    
    magma_d_matrix hx={Magma_CSR}, hy={Magma_CSR}, dx={Magma_CSR}, 
    dy={Magma_CSR}, hrefvec={Magma_CSR}, hcheck={Magma_CSR};
            
    hA_SELLP.blocksize = 8;
    hA_SELLP.alignment = 8;
    real_Double_t start, end, res;

    double c_one  = MAGMA_D_MAKE(1.0, 0.0);
    double c_zero = MAGMA_D_MAKE(0.0, 0.0);
    
    cusparseMatDescr_t descrA=NULL;
    cusparseHandle_t cusparseHandle = NULL;
    cusparseHybMat_t hybA=NULL;
    cusparseMatDescr_t descr = NULL;
    
    #ifdef MAGMA_WITH_MKL
        magma_int_t *pntre=NULL;
    #endif
    
    magma_int_t i, j;
    for( i = 1; i < argc; ++i ) {
        if ( strcmp("--blocksize", argv[i]) == 0 ) {
            hA_SELLP.blocksize = atoi( argv[++i] );
        } else if ( strcmp("--alignment", argv[i]) == 0 ) {
            hA_SELLP.alignment = atoi( argv[++i] );
        } else
            break;
    }
    printf( "\n#    usage: ./run_dspmv"
        " [ --blocksize %d --alignment %d (for SELLP) ]"
        " matrices \n\n", (int) hA_SELLP.blocksize, (int) hA_SELLP.alignment );

    while( i < argc ) {
        if ( strcmp("LAPLACE2D", argv[i]) == 0 && i+1 < argc ) {   // Laplace test
            i++;
            magma_int_t laplace_size = atoi( argv[i] );
            CHECK( magma_dm_5stencil(  laplace_size, &hA, queue ));
        } else {                        // file-matrix test
            CHECK( magma_d_csr_mtx( &hA,  argv[i], queue ));
        }

        printf( "\n# matrix info: %d-by-%d with %d nonzeros\n\n",
                            (int) hA.num_rows,(int) hA.num_cols,(int) hA.nnz );

        real_Double_t FLOPS = 2.0*hA.nnz/1e9;

        // init CPU vectors
        CHECK( magma_dvinit( &hx, Magma_CPU, hA.num_rows, 1, c_zero, queue ));
        CHECK( magma_dvinit( &hy, Magma_CPU, hA.num_rows, 1, c_zero, queue ));

        // init DEV vectors
        CHECK( magma_dvinit( &dx, Magma_DEV, hA.num_rows, 1, c_one, queue ));
        CHECK( magma_dvinit( &dy, Magma_DEV, hA.num_rows, 1, c_zero, queue ));
       

        #ifdef MAGMA_WITH_MKL
            // calling MKL with CSR
            CHECK( magma_imalloc_cpu( &pntre, hA.num_rows + 1 ) );
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
                mkl_dcsrmv( "N", &num_rows, &num_cols,
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
            row = NULL;
            col = NULL;
        #endif // MAGMA_WITH_MKL

        // copy matrix to GPU
        CHECK( magma_dmtransfer( hA, &dA, Magma_CPU, Magma_DEV, queue ));
        
        // warmup
        for (j=0; j<10; j++)
            CHECK( magma_d_spmv( c_one, dA, dx, c_zero, dy, queue ));

        // SpMV on GPU (CSR) -- this is the reference!
        start = magma_sync_wtime( queue );
        for (j=0; j<10; j++)
            CHECK( magma_d_spmv( c_one, dA, dx, c_zero, dy, queue ));
        end = magma_sync_wtime( queue );
        printf( " > MAGMA: %.2e seconds %.2e GFLOP/s    (standard CSR).\n",
                                        (end-start)/10, FLOPS*10/(end-start) );
        
        magma_dmfree(&dA, queue );
        CHECK( magma_dmtransfer( dy, &hrefvec , Magma_DEV, Magma_CPU, queue ));

        // convert to ELL and copy to GPU
        CHECK( magma_dmconvert(  hA, &hA_ELL, Magma_CSR, Magma_ELL, queue ));
        CHECK( magma_dmtransfer( hA_ELL, &dA_ELL, Magma_CPU, Magma_DEV, queue ));
        magma_dmfree(&hA_ELL, queue );
        magma_dmfree( &dy, queue );
        CHECK( magma_dvinit( &dy, Magma_DEV, hA.num_rows, 1, c_zero, queue ));
        // SpMV on GPU (ELL)
        start = magma_sync_wtime( queue );
        for (j=0; j<10; j++)
            CHECK( magma_d_spmv( c_one, dA_ELL, dx, c_zero, dy, queue ));
        end = magma_sync_wtime( queue );
        printf( " > MAGMA: %.2e seconds %.2e GFLOP/s    (standard ELL).\n",
                                        (end-start)/10, FLOPS*10/(end-start) );
        magma_dmfree(&dA_ELL, queue );
        CHECK( magma_dmtransfer( dy, &hcheck , Magma_DEV, Magma_CPU, queue ));
        res = 0.0;
        for(magma_int_t k=0; k<hA.num_rows; k++ )
            res=res + MAGMA_D_REAL(hcheck.val[k]) - MAGMA_D_REAL(hrefvec.val[k]);
        if ( res < .000001 )
            printf("%% tester spmv ELL:  ok\n");
        else
            printf("%% tester spmv ELL:  failed\n");
        magma_dmfree( &hcheck, queue );

        // convert to SELLP and copy to GPU
        CHECK( magma_dmconvert(  hA, &hA_SELLP, Magma_CSR, Magma_SELLP, queue ));
        CHECK( magma_dmtransfer( hA_SELLP, &dA_SELLP, Magma_CPU, Magma_DEV, queue ));
        magma_dmfree(&hA_SELLP, queue );
        magma_dmfree( &dy, queue );
        CHECK( magma_dvinit( &dy, Magma_DEV, hA.num_rows, 1, c_zero, queue ));
        // SpMV on GPU (SELLP)
        start = magma_sync_wtime( queue );
        for (j=0; j<10; j++)
            CHECK( magma_d_spmv( c_one, dA_SELLP, dx, c_zero, dy, queue ));
        end = magma_sync_wtime( queue );
        printf( " > MAGMA: %.2e seconds %.2e GFLOP/s    (SELLP).\n",
                                        (end-start)/10, FLOPS*10/(end-start) );

        CHECK( magma_dmtransfer( dy, &hcheck , Magma_DEV, Magma_CPU, queue ));
        res = 0.0;
        for(magma_int_t k=0; k<hA.num_rows; k++ )
            res=res + MAGMA_D_REAL(hcheck.val[k]) - MAGMA_D_REAL(hrefvec.val[k]);
        printf("%% |x-y|_F = %8.2e\n", res);
        if ( res < .000001 )
            printf("%% tester spmv SELL-P:  ok\n");
        else
            printf("%% tester spmv SELL-P:  failed\n");
        magma_dmfree( &hcheck, queue );

        magma_dmfree(&dA_SELLP, queue );


        // SpMV on GPU (CUSPARSE - CSR)
        // CUSPARSE context //

        CHECK_CUSPARSE( cusparseCreate( &cusparseHandle ));
        CHECK_CUSPARSE( cusparseSetStream( cusparseHandle, queue ));
        CHECK_CUSPARSE( cusparseCreateMatDescr( &descr ));

        CHECK_CUSPARSE( cusparseSetMatType( descr, CUSPARSE_MATRIX_TYPE_GENERAL ));
        CHECK_CUSPARSE( cusparseSetMatIndexBase( descr, CUSPARSE_INDEX_BASE_ZERO ));
        double alpha = c_one;
        double beta = c_zero;
        magma_dmfree( &dy, queue );
        CHECK( magma_dvinit( &dy, Magma_DEV, hA.num_rows, 1, c_zero, queue ));

        // copy matrix to GPU
        CHECK( magma_dmtransfer( hA, &dA, Magma_CPU, Magma_DEV, queue ));

        start = magma_sync_wtime( queue );
        for (j=0; j<10; j++)
            CHECK_CUSPARSE(
            cusparseDcsrmv(cusparseHandle,CUSPARSE_OPERATION_NON_TRANSPOSE,
                        hA.num_rows, hA.num_cols, hA.nnz, &alpha, descr,
                        dA.dval, dA.drow, dA.dcol, dx.dval, &beta, dy.dval) );
        end = magma_sync_wtime( queue );
        printf( " > CUSPARSE: %.2e seconds %.2e GFLOP/s    (CSR).\n",
                                        (end-start)/10, FLOPS*10/(end-start) );
        CHECK_CUSPARSE( cusparseCreateMatDescr( &descrA ));
        cusparseCreateHybMat( &hybA );
        CHECK( magma_dmtransfer( dy, &hcheck , Magma_DEV, Magma_CPU, queue ));
        res = 0.0;
        for(magma_int_t k=0; k<hA.num_rows; k++ )
            res=res + MAGMA_D_REAL(hcheck.val[k]) - MAGMA_D_REAL(hrefvec.val[k]);
        printf("%% |x-y|_F = %8.2e\n", res);
        if ( res < .000001 )
            printf("%% tester spmv cuSPARSE CSR:  ok\n");
        else
            printf("%% tester spmv cuSPARSE CSR:  failed\n");
        magma_dmfree( &hcheck, queue );
        magma_dmfree( &dy, queue );
        CHECK( magma_dvinit( &dy, Magma_DEV, hA.num_rows, 1, c_zero, queue ));
       
        cusparseDcsr2hyb(cusparseHandle,  hA.num_rows, hA.num_cols,
                        descrA, dA.dval, dA.drow, dA.dcol,
                        hybA, 0, CUSPARSE_HYB_PARTITION_AUTO);

        start = magma_sync_wtime( queue );
        for (j=0; j<10; j++)
            CHECK_CUSPARSE(
            cusparseDhybmv( cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
               &alpha, descrA, hybA,
               dx.dval, &beta, dy.dval) );
        end = magma_sync_wtime( queue );
        printf( " > CUSPARSE: %.2e seconds %.2e GFLOP/s    (HYB).\n",
                                        (end-start)/10, FLOPS*10/(end-start) );

        CHECK( magma_dmtransfer( dy, &hcheck , Magma_DEV, Magma_CPU, queue ));
        res = 0.0;
        for(magma_int_t k=0; k<hA.num_rows; k++ )
            res=res + MAGMA_D_REAL(hcheck.val[k]) - MAGMA_D_REAL(hrefvec.val[k]);
        printf("%% |x-y|_F = %8.2e\n", res);
        if ( res < .000001 )
            printf("%% tester spmv cuSPARSE HYB:  ok\n");
        else
            printf("%% tester spmv cuSPARSE HYB:  failed\n");
        magma_dmfree( &hcheck, queue );

        cusparseDestroyMatDescr( descrA );
        cusparseDestroyHybMat( hybA );
        cusparseDestroy( cusparseHandle ); 
        descrA=NULL;
        cusparseHandle = NULL;
        hybA=NULL;
        descr = NULL;

        // free CPU memory
        magma_dmfree(&hA, queue );
        magma_dmfree(&hx, queue );
        magma_dmfree(&hy, queue );
        magma_dmfree(&hrefvec, queue );
        // free GPU memory
        magma_dmfree(&dA, queue );
        magma_dmfree(&dx, queue );
        magma_dmfree(&dy, queue );
        
        printf("\n\n");

        i++;
    }
    
cleanup:
    #ifdef MAGMA_WITH_MKL
        magma_free_cpu(pntre);
    #endif    
    cusparseDestroyMatDescr( descrA );
    cusparseDestroyHybMat( hybA );
    cusparseDestroy( cusparseHandle ); 
    magma_dmfree(&hA, queue );
    magma_dmfree(&dA, queue );
    magma_dmfree(&hA_ELL, queue );
    magma_dmfree(&dA_ELL, queue );
    magma_dmfree(&hA_SELLP, queue );
    magma_dmfree(&dA_SELLP, queue );
    magma_dmfree( &hcheck, queue );
    magma_dmfree(&hx, queue );
    magma_dmfree(&hy, queue );
    magma_dmfree(&hrefvec, queue );
    magma_dmfree(&dx, queue );
    magma_dmfree(&dy, queue );
    
    magma_queue_destroy( queue );
    TESTING_FINALIZE();
    return info;
}
