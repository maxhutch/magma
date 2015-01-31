/*
    -- MAGMA (version 1.6.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date January 2015

       @generated from testing_zspmm.cpp normal z -> d, Fri Jan 30 19:00:33 2015
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
    #include "mkl_spblas.h"
    
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
    
    magma_d_sparse_matrix hA, hA_SELLP, hA_ELL, dA, dA_SELLP, dA_ELL;
    hA_SELLP.blocksize = 8;
    hA_SELLP.alignment = 8;
    real_Double_t start, end, res;
    magma_int_t *pntre;

    double c_one  = MAGMA_D_MAKE(1.0, 0.0);
    double c_zero = MAGMA_D_MAKE(0.0, 0.0);
    
    magma_int_t i, j;
    for( i = 1; i < argc; ++i ) {
        if ( strcmp("--blocksize", argv[i]) == 0 ) {
            hA_SELLP.blocksize = atoi( argv[++i] );
        } else if ( strcmp("--alignment", argv[i]) == 0 ) {
            hA_SELLP.alignment = atoi( argv[++i] );
        } else
            break;
    }
    printf( "\n#    usage: ./run_dspmm"
        " [ --blocksize %d --alignment %d (for SELLP) ]"
        " matrices \n\n", (int) hA_SELLP.blocksize, (int) hA_SELLP.alignment );

    while(  i < argc ) {

        if ( strcmp("LAPLACE2D", argv[i]) == 0 && i+1 < argc ) {   // Laplace test
            i++;
            magma_int_t laplace_size = atoi( argv[i] );
            magma_dm_5stencil(  laplace_size, &hA, queue );
        } else {                        // file-matrix test
            magma_d_csr_mtx( &hA,  argv[i], queue );
        }

        printf( "# matrix info: %d-by-%d with %d nonzeros\n",
                            (int) hA.num_rows,(int) hA.num_cols,(int) hA.nnz );

        real_Double_t FLOPS = 2.0*hA.nnz/1e9;

        magma_d_vector hx, hy, dx, dy, hrefvec, hcheck;

        // m - number of rows for the sparse matrix
        // n - number of vectors to be multiplied in the SpMM product
        magma_int_t m, n;

        m = hA.num_rows;
        n = 48;

        // init CPU vectors
        magma_d_vinit( &hx, Magma_CPU, m*n, c_one, queue );
        magma_d_vinit( &hy, Magma_CPU, m*n, c_zero, queue );

        // init DEV vectors
        magma_d_vinit( &dx, Magma_DEV, m*n, c_one, queue );
        magma_d_vinit( &dy, Magma_DEV, m*n, c_zero, queue );


        // calling MKL with CSR
        pntre = (magma_int_t*)malloc( (m+1)*sizeof(magma_int_t) );
        pntre[0] = 0;
        for (j=0; j<m; j++ ) pntre[j] = hA.row[j+1];

        #ifdef MAGMA_WITH_MKL
            MKL_INT num_rows = hA.num_rows;
            MKL_INT num_cols = hA.num_cols;
            MKL_INT nnz = hA.nnz;
            MKL_INT num_vecs = n;

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

            // === Call MKL with consecutive SpMVs, using mkl_dcsrmv ===
            // warmp up
            mkl_dcsrmv( "N", &num_rows, &num_cols, 
                        MKL_ADDR(&c_one), "GFNC", MKL_ADDR(hA.val), col, row, pntre,
                                                MKL_ADDR(hx.val),
                        MKL_ADDR(&c_zero),        MKL_ADDR(hy.val) );
    
            start = magma_wtime(); 
            for (j=0; j<10; j++ )
                mkl_dcsrmv( "N", &num_rows, &num_cols, 
                        MKL_ADDR(&c_one), "GFNC", MKL_ADDR(hA.val), col, row, pntre,
                                                MKL_ADDR(hx.val),
                        MKL_ADDR(&c_zero),        MKL_ADDR(hy.val) );
            end = magma_wtime();
            printf( "\n > MKL SpMVs : %.2e seconds %.2e GFLOP/s    (CSR).\n",
                                            (end-start)/10, FLOPS*10/(end-start) );
    
            // === Call MKL with blocked SpMVs, using mkl_dcsrmm ===
            char transa = 'n';
            MKL_INT ldb = n, ldc=n;
            char matdescra[6] = {'g', 'l', 'n', 'c', 'x', 'x'};
    
            // warm up
            mkl_dcsrmm( &transa, &num_rows, &num_vecs, &num_cols, MKL_ADDR(&c_one), matdescra, 
                      MKL_ADDR(hA.val), col, row, pntre,
                      MKL_ADDR(hx.val), &ldb,
                      MKL_ADDR(&c_zero),        
                      MKL_ADDR(hy.val), &ldc );
    
            start = magma_wtime();
            for (j=0; j<10; j++ )
                mkl_dcsrmm( &transa, &num_rows, &num_vecs, &num_cols, MKL_ADDR(&c_one), matdescra, 
                          MKL_ADDR(hA.val), col, row, pntre,
                          MKL_ADDR(hx.val), &ldb,
                          MKL_ADDR(&c_zero),        
                          MKL_ADDR(hy.val), &ldc );
            end = magma_wtime();
            printf( "\n > MKL SpMM  : %.2e seconds %.2e GFLOP/s    (CSR).\n",
                    (end-start)/10, FLOPS*10.*n/(end-start) );

            TESTING_FREE_CPU( row );
            TESTING_FREE_CPU( col );
            free(pntre);
        #endif // MAGMA_WITH_MKL

        // copy matrix to GPU
        magma_d_mtransfer( hA, &dA, Magma_CPU, Magma_DEV, queue );
        // SpMV on GPU (CSR)
        start = magma_sync_wtime( queue ); 
        for (j=0; j<10; j++)
            magma_d_spmv( c_one, dA, dx, c_zero, dy, queue );
        end = magma_sync_wtime( queue ); 
        printf( " > MAGMA: %.2e seconds %.2e GFLOP/s    (standard CSR).\n",
                                        (end-start)/10, FLOPS*10.*n/(end-start) );

        magma_d_vtransfer( dy, &hrefvec , Magma_DEV, Magma_CPU, queue );
        magma_d_mfree(&dA, queue );

        // convert to ELL and copy to GPU
        magma_d_mconvert(  hA, &hA_ELL, Magma_CSR, Magma_ELL, queue );
        magma_d_mtransfer( hA_ELL, &dA_ELL, Magma_CPU, Magma_DEV, queue );
        magma_d_mfree(&hA_ELL, queue );
        magma_d_vfree( &dy, queue );
        magma_d_vinit( &dy, Magma_DEV, dx.num_rows, c_zero, queue );
        // SpMV on GPU (ELL)
        start = magma_sync_wtime( queue ); 
        for (j=0; j<10; j++)
            magma_d_spmv( c_one, dA_ELL, dx, c_zero, dy, queue );
        end = magma_sync_wtime( queue ); 
        printf( " > MAGMA: %.2e seconds %.2e GFLOP/s    (standard ELL).\n",
                                        (end-start)/10, FLOPS*10.*n/(end-start) );

        magma_d_vtransfer( dy, &hcheck , Magma_DEV, Magma_CPU, queue );
        res = 0.0;
        for(magma_int_t k=0; k<hA.num_rows; k++ )
            res=res + MAGMA_D_REAL(hcheck.val[k]) - MAGMA_D_REAL(hrefvec.val[k]);
        printf("# |x-y|_F = %8.2e\n", res);
        if ( res < .000001 )
            printf("# tester spmm ELL:  ok\n");
        else
            printf("# tester spmm ELL:  failed\n");
        magma_d_vfree( &hcheck, queue );

        magma_d_mfree(&dA_ELL, queue );


        // convert to SELLP and copy to GPU
        magma_d_mconvert(  hA, &hA_SELLP, Magma_CSR, Magma_SELLP, queue );
        magma_d_mtransfer( hA_SELLP, &dA_SELLP, Magma_CPU, Magma_DEV, queue );
        magma_d_mfree(&hA_SELLP, queue );
        magma_d_vfree( &dy, queue );
        magma_d_vinit( &dy, Magma_DEV, dx.num_rows, c_zero, queue );
        // SpMV on GPU (SELLP)
        start = magma_sync_wtime( queue ); 
        for (j=0; j<10; j++)
            magma_d_spmv( c_one, dA_SELLP, dx, c_zero, dy, queue );
        end = magma_sync_wtime( queue ); 
        printf( " > MAGMA: %.2e seconds %.2e GFLOP/s    (SELLP).\n",
                                        (end-start)/10, FLOPS*10.*n/(end-start) );

        magma_d_vtransfer( dy, &hcheck , Magma_DEV, Magma_CPU, queue );
        res = 0.0;
        for(magma_int_t k=0; k<hA.num_rows; k++ )
            res=res + MAGMA_D_REAL(hcheck.val[k]) - MAGMA_D_REAL(hrefvec.val[k]);
        printf("# |x-y|_F = %8.2e\n", res);
        if ( res < .000001 )
            printf("# tester spmm SELL-P:  ok\n");
        else
            printf("# tester spmm SELL-P:  failed\n");
        magma_d_vfree( &hcheck, queue );
        magma_d_mfree(&dA_SELLP, queue );



        // SpMV on GPU (CUSPARSE - CSR)
        // CUSPARSE context //
        magma_d_vfree( &dy, queue );
        magma_d_vinit( &dy, Magma_DEV, dx.num_rows, c_zero, queue );
        #ifdef PRECISION_d
        start = magma_sync_wtime( queue ); 
        cusparseHandle_t cusparseHandle = 0;
        cusparseStatus_t cusparseStatus;
        cusparseStatus = cusparseCreate(&cusparseHandle);
        cusparseSetStream( cusparseHandle, queue );

        cusparseMatDescr_t descr = 0;
        cusparseStatus = cusparseCreateMatDescr(&descr);

        cusparseSetMatType(descr,CUSPARSE_MATRIX_TYPE_GENERAL);
        cusparseSetMatIndexBase(descr,CUSPARSE_INDEX_BASE_ZERO);
        double alpha = c_one;
        double beta = c_zero;

        // copy matrix to GPU
        magma_d_mtransfer( hA, &dA, Magma_CPU, Magma_DEV);

        for (j=0; j<10; j++)                    
        cusparseDcsrmm(cusparseHandle,
            CUSPARSE_OPERATION_NON_TRANSPOSE, 
                    dA.num_rows,   n, dA.num_cols, dA.nnz, 
                    &alpha, descr, dA.dval, dA.drow, dA.dcol,
                    dx.dval, dA.num_cols, &beta, dy.dval, dA.num_cols);
        end = magma_sync_wtime( queue ); 
        printf( " > CUSPARSE: %.2e seconds %.2e GFLOP/s    (CSR).\n",
                                        (end-start)/10, FLOPS*10*n/(end-start) );

        magma_d_vtransfer( dy, &hcheck , Magma_DEV, Magma_CPU, queue );
        res = 0.0;
        for(magma_int_t k=0; k<hA.num_rows; k++ )
            res=res + MAGMA_D_REAL(hcheck.val[k]) - MAGMA_D_REAL(hrefvec.val[k]);
        printf("# |x-y|_F = %8.2e\n", res);
        if ( res < .000001 )
            printf("# tester spmm cuSPARSE:  ok\n");
        else
            printf("# tester spmm cuSPARSE:  failed\n");
        magma_d_vfree( &hcheck, queue );

        cusparseDestroyMatDescr( descr );
        cusparseDestroy( cusparseHandle );

        magma_d_mfree(&dA);

        #endif

        printf("\n\n");


        // free CPU memory
        magma_d_mfree(&hA, queue );
        magma_d_vfree(&hx, queue );
        magma_d_vfree(&hy, queue );
        magma_d_vfree(&hrefvec, queue );
        // free GPU memory
        magma_d_vfree(&dx, queue );
        magma_d_vfree(&dy, queue );

        i++;

    }

    magma_queue_destroy( queue );
    TESTING_FINALIZE();
    return 0;
}
