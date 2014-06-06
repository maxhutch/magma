/*
    -- MAGMA (version 1.5.0-beta2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date May 2014

       @author Hartwig Anzt 

       @generated from zbcsrlu.cpp normal z -> d, Fri May 30 10:41:41 2014
*/

#include <cuda_runtime_api.h>
#include <cublas_v2.h>  // include before magma.h

#include "magma.h"
#include "magma_lapack.h"
#include <stdio.h>
#include <stdlib.h>


//#include "common_magma.h"
#include "../include/magmasparse.h"

#define PRECISION_d

#define  blockinfo(i,j)  A.blockinfo[(i)*c_blocks   + (j)]
#define  Mblockinfo(i,j)  M->blockinfo[(i)*c_blocks   + (j)]
#define M(i,j) M->val+((Mblockinfo(i,j)-1)*size_b*size_b)
#define A(i,j) A.val+((blockinfo(i,j)-1)*size_b*size_b)
#define x(i) x->val+(i*size_b)

#ifndef max
#define max(a, b) ((a) > (b) ? (a) : (b))
#endif

#define RTOLERANCE     lapackf77_dlamch( "E" )
#define ATOLERANCE     lapackf77_dlamch( "E" )


/*  -- MAGMA (version 1.5.0-beta2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date May 2014

    Purpose
    =======

    LU decomposition and solution via triangular solves for a BCSR matrix A. 
    We assume all diagonal blocks to be nonzero.

    Arguments
    =========

    magma_d_sparse_matrix A                   descriptor for matrix A
    magma_d_vector b                          RHS b vector
    magma_d_vector *x                         solution approximation
    magma_d_solver_par *solver_par       solver parameters

    ========================================================================  */

magma_int_t
magma_dbcsrlu( magma_d_sparse_matrix A, magma_d_vector b, 
                       magma_d_vector *x, magma_d_solver_par *solver_par ){

    // prepare solver feedback
    solver_par->solver = Magma_BCSRLU;
    real_Double_t t_lu1, t_lu = 0.0;
    real_Double_t t_lusv1, t_lusv = 0.0;
    double residual;
    magma_d_sparse_matrix A_d;
    magma_d_mtransfer( A, &A_d, Magma_CPU, Magma_DEV);
    magma_dresidual( A_d, b, *x, &residual );
    solver_par->init_res = residual;
        magma_malloc_cpu( (void **)&solver_par->timing, 
                                    2*sizeof(real_Double_t) );
    solver_par->res_vec = NULL;



    if( A.memory_location == Magma_DEV){
        magma_d_sparse_matrix B;
        magma_d_mtransfer( A, &B, Magma_DEV, Magma_CPU ); 
        magma_dbcsrlu( B,  b, x, solver_par );
        magma_d_mfree(&B);
        return MAGMA_SUCCESS;
    }
    else{

    magma_d_sparse_matrix B, C, D;
    // compute suitable blocksize
    for(int defaultsize=400; defaultsize>=16; defaultsize--){
        if( A.num_rows%defaultsize == 0 )
            B.blocksize = defaultsize;
    }

    magma_d_mconvert( A, &B, Magma_CSR, Magma_BCSR);
    magma_d_mtransfer( B, &C, Magma_CPU, Magma_DEV);
    // pivot array for local pivoting
    magma_int_t *ipiv;
    magma_imalloc_cpu( &ipiv, C.blocksize
                *(ceil( (float)C.num_rows / (float)C.blocksize )+1) );

    // LU factorization
    magma_device_sync(); t_lu1=magma_wtime();
    magma_dbcsrlutrf( C, &D, ipiv, solver_par->version );
    magma_device_sync(); t_lu+=(magma_wtime()-t_lu1);


    // triangular solves
    magma_device_sync(); t_lusv1=magma_wtime();
    magma_dbcsrlusv( D, b, x, solver_par, ipiv );
    magma_device_sync(); t_lusv+=(magma_wtime()-t_lusv1);


    magma_d_mfree(&B);
    magma_d_mfree(&C);
    magma_d_mfree(&D);
    magma_free_cpu(ipiv);

    solver_par->timing[0] = (real_Double_t)t_lu;
    solver_par->timing[1] = (real_Double_t)t_lusv;
    solver_par->runtime = t_lu+t_lusv;

    magma_dresidual( A_d, b, *x, &residual );
    solver_par->final_res = residual;

    magma_d_mfree(&A_d);

     if( solver_par->init_res > solver_par->final_res )
        solver_par->info = 0;
    else
        solver_par->info = -1;

    return MAGMA_SUCCESS;
    }
}   /* magma_dbcsrlu */
    





/*  -- MAGMA (version 1.5.0-beta2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date May 2014

    Purpose
    =======

    LU decomposition for a BCSR matrix A. 
    We assume all diagonal blocks to be nonzero.

    Arguments
    =========

    magma_d_sparse_matrix A           input matrix A (on DEV)
    magma_d_sparse_matrix *M          output matrix containing LU decomposition
    magma_int_t *ipiv                 pivot vector

    ========================================================================  */

magma_int_t
magma_dbcsrlutrf( magma_d_sparse_matrix A, magma_d_sparse_matrix *M, 
                                       magma_int_t *ipiv, magma_int_t version ){


    // some useful variables
    double one = MAGMA_D_ONE;
    double m_one = MAGMA_D_NEG_ONE;
    magma_int_t i,j,k, info;

    cublasHandle_t handle;
    cudaSetDevice( 0 );
    cublasCreate( &handle );


    // GPU stream
    const int num_streams = 16;
    magma_queue_t stream[num_streams];
    for( i=0; i<num_streams; i++ )
        magma_queue_create( &stream[i] );
    // fill in information for B
    M->storage_type = A.storage_type;
    M->memory_location = Magma_DEV;
    M->num_rows = A.num_rows;
    M->num_cols = A.num_cols;
    M->blocksize = A.blocksize;
    magma_int_t size_b = A.blocksize;
    magma_int_t c_blocks = ceil( (float)A.num_cols / (float)size_b );     
                // max number of blocks per row
    magma_int_t r_blocks = ceil( (float)A.num_rows / (float)size_b );     
                // max number of blocks per column

    //complete fill-in
    magma_indexmalloc_cpu( &M->blockinfo, r_blocks * c_blocks );

    for( k=0; k<r_blocks; k++){
        for( j=0; j<c_blocks; j++ ){
            Mblockinfo(k,j) = blockinfo(k,j);
        }
    }
    for( k=0; k<r_blocks; k++){
        for( j=k+1; j<r_blocks; j++ ){
            if( (Mblockinfo(j,k)!=0) ){
                for( i=k+1; i<c_blocks; i++ ){
                    if( (Mblockinfo(j,i)==0) && (Mblockinfo(k,i)!=0) ){
                        Mblockinfo(j,i) = -1;
                    }
                }
            }
        }
    }  
    magma_int_t num_blocks_tmp = 0;
    for( magma_int_t  il=0; il<r_blocks * c_blocks; il++ ){
        if( M->blockinfo[il]!=0 ){
            num_blocks_tmp++;
            M->blockinfo[il] = num_blocks_tmp;
        }
    }
    M->numblocks = num_blocks_tmp;
    magma_dmalloc( &M->val, size_b*size_b*(M->numblocks) );
    magma_indexmalloc( &M->row, r_blocks+1 );
    magma_indexmalloc( &M->col, M->numblocks );

    // Prepare A 
    double **hA, **dA,  **hB, **dB, **hBL, **dBL, **hC, **dC;
    int rowt=0, rowt2=0;
    magma_malloc_cpu((void **)& hA, (M->numblocks)
                            *sizeof(double *) );
    magma_malloc(    (void **)&dA, (M->numblocks)
                            *sizeof(double *) );
    magma_malloc_cpu((void **)& hB, (M->numblocks)
                            *sizeof(double *) );
    magma_malloc(    (void **)&dB, (M->numblocks)
                            *sizeof(double *) );
    magma_malloc_cpu((void **)& hBL, (M->numblocks)
                            *sizeof(double *) );
    magma_malloc(    (void **)&dBL, (M->numblocks)
                            *sizeof(double *) );
    magma_malloc_cpu((void **)& hC, (M->numblocks)
                            *sizeof(double *) );
    magma_malloc(    (void **)&dC, (M->numblocks)
                            *sizeof(double *) );

    double **AIs, **AIIs, **dAIs, **dAIIs, **BIs, **dBIs;

    magma_malloc_cpu((void **)&AIs, r_blocks*c_blocks
                        *sizeof(double *));
    magma_malloc_cpu((void **)&BIs, r_blocks*c_blocks
                        *sizeof(double *));
    magma_malloc((void **)&dAIs, r_blocks*c_blocks
                        *sizeof(double *));
    magma_malloc((void **)&dBIs, r_blocks*c_blocks
                        *sizeof(double *));
    magma_malloc_cpu((void **)&AIIs, r_blocks*c_blocks
                        *sizeof(double *));
    magma_malloc((void **)&dAIIs, r_blocks*c_blocks
                        *sizeof(double *));

    magma_int_t *ipiv_d;
    magma_imalloc( &ipiv_d, size_b);
    for( i = 0; i< r_blocks; i++){
        for( j = 0; j< r_blocks; j++){
           if ( (Mblockinfo(i, j) != 0) && (blockinfo(i,j)!=0) ){
              hA[rowt] = A(i, j);
              hB[rowt] = M(i, j);
              rowt++;
           }
           else if ( (Mblockinfo(i, j) != 0) && (blockinfo(i, j) == 0) ){
              hC[rowt2] = M(i, j);
              rowt2++;
           }
        }
    }
    cublasSetVector(  A.numblocks, sizeof(double *), 
                                                        hA, 1, dA, 1 );
    cublasSetVector(  A.numblocks, sizeof(double *), 
                                                        hB, 1, dB, 1 );
    cublasSetVector(  (M->numblocks-A.numblocks), 
                        sizeof(double *), hC, 1, dC, 1 );

    magma_dbcsrvalcpy(  size_b, A.numblocks, (M->numblocks-A.numblocks), 
                                                        dA, dB, dC );

    num_blocks_tmp=0;
    magma_index_t *cpu_row, *cpu_col;
    magma_indexmalloc_cpu( &cpu_row, r_blocks+1 );
    magma_indexmalloc_cpu( &cpu_col, M->numblocks );

    num_blocks_tmp=0;
    for( i=0; i<c_blocks * r_blocks; i++ ){
        if( i%c_blocks == 0) {
            magma_int_t tmp = i/c_blocks;
            cpu_row[tmp] = num_blocks_tmp;

        }
        if( M->blockinfo[i] != 0 ){
            magma_int_t tmp = i%c_blocks;
            cpu_col[num_blocks_tmp] = tmp;

            num_blocks_tmp++;
        }
    }
    cpu_row[r_blocks] = num_blocks_tmp;
    M->nnz = num_blocks_tmp;

    cublasSetVector( r_blocks+1, sizeof( magma_int_t ), cpu_row, 1, M->row, 1 );            
    cublasSetVector(  M->numblocks, sizeof( magma_int_t ), cpu_col, 
                                                               1, M->col, 1 );
    magma_free_cpu( cpu_row );
    magma_free_cpu( cpu_col );

    magma_int_t ldda, lddb, lddc, ldwork;

    ldda = size_b;//((size_b+31)/32)*32;
    lddb = size_b;//((size_b+31)/32)*32;
    lddc = size_b;//((size_b+31)/32)*32;

    double *dwork;
    ldwork = size_b * magma_get_dgetri_nb( size_b );
    magma_dmalloc( &dwork, ldwork );

    //--------------------------------------------------------------------------
    //  LU factorization
    // kij-version
    for( k=0; k<r_blocks; k++){

        int num_block_rows = 0, kblocks = 0, klblocks = 0, 
                                row = 0, row1 = 0, row2 = 0, row3 = 0;
        for( i = 0; i< k; i++){
            if ( Mblockinfo( k, i ) != 0 ){
                klblocks++;
                hBL[row3] = M( k, i);
                row3++;
            }
        }
        cublasSetVector( klblocks, sizeof(double *), hBL, 1, 
                                                                    dBL, 1 );

        for( i = k+1; i< c_blocks; i++){
            if ( Mblockinfo( k, i ) != 0 ){
              kblocks++;
              hB[row1] = M( k, i);
              row1++;

            }
            if ( Mblockinfo(i , k ) != 0 ){
                num_block_rows++;
                hA[row2] =M(i, k);
                row2++;
            }
        }
        cublasSetVector( num_block_rows, sizeof(double *), hA, 1, 
                                                        dA, 1 );
        cublasSetVector( kblocks, sizeof(double *), hB, 1, dB, 1 );
            
        for( i = k+1; i< r_blocks; i++){
           if ( Mblockinfo(i, k) != 0 ){
              for( j = k+1; j<c_blocks; j++ ){
                 if ( Mblockinfo(k, j) != 0 ){
                    hC[row] = M(i, j);
                    row++;
                 }
              }
           }
        }
        cublasSetVector( kblocks*num_block_rows, sizeof(double *), 
                                                                hC, 1, dC, 1 );

        if( version==0 ){ 
        // AIs and BIs for the batched GEMMs later
            for(i=0; i<num_block_rows; i++){
               for(j=0; j<kblocks; j++){
                  AIs[j+i*kblocks] = hA[i];
                  BIs[j+i*kblocks] = hB[j];
                }
            }
            cublasSetVector( kblocks*num_block_rows, 
                        sizeof(double *), AIs, 1, dAIs, 1 );
            cublasSetVector( kblocks*num_block_rows, 
                        sizeof(double *), BIs, 1, dBIs, 1 );
        }  
        // AIIs for the batched TRSMs under the factorized block
        for(i=0; i<max(num_block_rows, kblocks); i++){
           AIIs[i] = M(k,k);
        }
        cublasSetVector( max(num_block_rows, kblocks), 
                sizeof(double *), AIIs, 1, dAIIs, 1 );
        
        magma_dgetrf_gpu( size_b, size_b, M(k,k), ldda, ipiv+k*size_b, &info );


        // Swap elements on the right before update
        cublasSetVector( size_b, sizeof( int ), ipiv+k*size_b, 1, ipiv_d, 1 ); 
        magma_dbcsrlupivloc( size_b, kblocks, dB, ipiv_d );


        // update blocks right
        cublasDtrsmBatched( handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER, 
                            CUBLAS_OP_N, CUBLAS_DIAG_UNIT, size_b, size_b, 
                            &one, dAIIs, size_b, dB, size_b, kblocks );



        // Swap elements on the left (anytime - hidden by the Ztrsm)
        magmablasSetKernelStream( stream[1] );
        magma_dbcsrlupivloc( size_b, klblocks, dBL, ipiv_d );



        // update blocks below
        cublasDtrsmBatched( handle, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_UPPER, 
                            CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, size_b, size_b, 
                            &one, dAIIs, size_b, dA, size_b, num_block_rows );
        
        if( version==1 ){ 
            magmablasSetKernelStream( stream[0] );
            magma_dbcsrluegemm( size_b, num_block_rows, kblocks, dA, dB, dC ); 
        }

        if( version==0 ){ 
            //------------------------------------------------------------------
            // update trailing matrix using cublas batched GEMM
            magmablasSetKernelStream( stream[0] );
            cublasDgemmBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N, size_b, size_b, 
                              size_b,&m_one, (const double **) dAIs,  
                              size_b, (const double **) dBIs, 
                              size_b, &one, dC , size_b, 
                              kblocks*num_block_rows );
            // end update trailing matrix using cublas batched GEMM
            //------------------------------------------------------------------
        }
    
    }// end block k

    magma_free( dA );
    magma_free( dB );
    magma_free( dBL );
    magma_free( dC );
    magma_free_cpu( hA );
    magma_free_cpu( hB );
    magma_free_cpu( hBL );
    magma_free_cpu( hC );


    magma_free( dBIs );
    magma_free( dAIIs );
    magma_free_cpu( AIIs );
    magma_free_cpu( BIs );
    magma_free_cpu( AIs );
    magma_free( dAIs );

    return MAGMA_SUCCESS;
}   /* magma_dbcsrlusv */




magma_int_t
magma_dbcsrlusv( magma_d_sparse_matrix A, magma_d_vector b, magma_d_vector *x,  
           magma_d_solver_par *solver_par, magma_int_t *ipiv ){

    // some useful variables
    magma_int_t size_b = A.blocksize;
    magma_int_t c_blocks = ceil( (float)A.num_cols / (float)size_b );     
                                        // max number of blocks per row
    magma_int_t r_blocks = ceil( (float)A.num_rows / (float)size_b );     
                                        // max number of blocks per column

    // set x = b
    cudaMemcpy( x->val, b.val, A.num_rows*sizeof( double ), 
                                            cudaMemcpyDeviceToDevice );

    // First pivot the RHS
    magma_dbcsrswp( r_blocks, size_b, ipiv, x->val );


    // forward solve
    magma_dbcsrtrsv( MagmaLower, r_blocks, c_blocks, size_b, 
                     A.val, A.blockinfo, x->val );

    // backward solve
    magma_dbcsrtrsv( MagmaUpper, r_blocks, c_blocks, size_b, 
                     A.val, A.blockinfo, x->val );

    return MAGMA_SUCCESS;
}   /* magma_dbcsrlusv */



