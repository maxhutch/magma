/*
    -- MAGMA (version 1.5.0-beta3) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date July 2014

       @author Hartwig Anzt 

       @generated from zbcsrlu.cpp normal z -> s, Fri Jul 18 17:34:29 2014
*/

#include <cuda_runtime_api.h>
#include <cublas_v2.h>  // include before magma.h

#include "magma.h"
#include "magma_lapack.h"
#include <stdio.h>
#include <stdlib.h>


//#include "common_magma.h"
#include "../include/magmasparse.h"

#define PRECISION_s

#define  blockinfo(i,j)  A.blockinfo[(i)*c_blocks   + (j)]
#define  Mblockinfo(i,j)  M->blockinfo[(i)*c_blocks   + (j)]
#define M(i,j) M->val+((Mblockinfo(i,j)-1)*size_b*size_b)
#define A(i,j) A.val+((blockinfo(i,j)-1)*size_b*size_b)
#define x(i) x->val+(i*size_b)

#ifndef max
#define max(a, b) ((a) > (b) ? (a) : (b))
#endif

#define RTOLERANCE     lapackf77_slamch( "E" )
#define ATOLERANCE     lapackf77_slamch( "E" )


/**
    Purpose
    -------

    LU decomposition and solution via triangular solves for a BCSR matrix A. 
    We assume all diagonal blocks to be nonzero.

    Arguments
    ---------

    @param
    A           magma_s_sparse_matrix
                descriptor for matrix A

    @param
    b           magma_s_vector
                RHS b vector

    @param
    x           magma_s_vector*
                solution approximation

    @param
    solver_par  magma_s_solver_par*
                solver parameters

    @ingroup magmasparse_sgesv
    ********************************************************************/

magma_int_t
magma_sbcsrlu( magma_s_sparse_matrix A, magma_s_vector b, 
                       magma_s_vector *x, magma_s_solver_par *solver_par ){

    // prepare solver feedback
    solver_par->solver = Magma_BCSRLU;
    real_Double_t t_lu1, t_lu = 0.0;
    real_Double_t t_lusv1, t_lusv = 0.0;
    float residual;
    magma_s_sparse_matrix A_d;
    magma_s_mtransfer( A, &A_d, Magma_CPU, Magma_DEV);
    magma_sresidual( A_d, b, *x, &residual );
    solver_par->init_res = residual;
        magma_malloc_cpu( (void **)&solver_par->timing, 
                                    2*sizeof(real_Double_t) );
    solver_par->res_vec = NULL;



    if( A.memory_location == Magma_DEV){
        magma_s_sparse_matrix B;
        magma_s_mtransfer( A, &B, Magma_DEV, Magma_CPU ); 
        magma_sbcsrlu( B,  b, x, solver_par );
        magma_s_mfree(&B);
        return MAGMA_SUCCESS;
    }
    else{

    magma_s_sparse_matrix B, C, D;
    // compute suitable blocksize
    for(int defaultsize=400; defaultsize>=16; defaultsize--){
        if( A.num_rows%defaultsize == 0 )
            B.blocksize = defaultsize;
    }

    magma_s_mconvert( A, &B, Magma_CSR, Magma_BCSR);
    magma_s_mtransfer( B, &C, Magma_CPU, Magma_DEV);
    // pivot array for local pivoting
    magma_int_t *ipiv;
    magma_imalloc_cpu( &ipiv, C.blocksize
                *(ceil( (float)C.num_rows / (float)C.blocksize )+1) );

    // LU factorization
    magma_device_sync(); t_lu1=magma_wtime();
    magma_sbcsrlutrf( C, &D, ipiv, solver_par->version );
    magma_device_sync(); t_lu+=(magma_wtime()-t_lu1);


    // triangular solves
    magma_device_sync(); t_lusv1=magma_wtime();
    magma_sbcsrlusv( D, b, x, solver_par, ipiv );
    magma_device_sync(); t_lusv+=(magma_wtime()-t_lusv1);


    magma_s_mfree(&B);
    magma_s_mfree(&C);
    magma_s_mfree(&D);
    magma_free_cpu(ipiv);

    solver_par->timing[0] = (real_Double_t)t_lu;
    solver_par->timing[1] = (real_Double_t)t_lusv;
    solver_par->runtime = t_lu+t_lusv;

    magma_sresidual( A_d, b, *x, &residual );
    solver_par->final_res = residual;

    magma_s_mfree(&A_d);

     if( solver_par->init_res > solver_par->final_res )
        solver_par->info = 0;
    else
        solver_par->info = -1;

    return MAGMA_SUCCESS;
    }
}   /* magma_sbcsrlu */
    





/**
    Purpose
    -------

    LU decomposition for a BCSR matrix A. 
    We assume all diagonal blocks to be nonzero.

    Arguments
    ---------

    @param
    A           magma_s_sparse_matrix
                input matrix A (on DEV)

    @param
    m           magma_s_sparse_matrix*
                output matrix containing LU decomposition

    @param
    ipiv        magma_int_t*
                pivot vector


    @ingroup magmasparse_sgesv
    ********************************************************************/

magma_int_t
magma_sbcsrlutrf( magma_s_sparse_matrix A, magma_s_sparse_matrix *M, 
                                       magma_int_t *ipiv, magma_int_t version ){


    // some useful variables
    float one = MAGMA_S_ONE;
    float m_one = MAGMA_S_NEG_ONE;
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
    magma_index_malloc_cpu( &M->blockinfo, r_blocks * c_blocks );

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
    magma_smalloc( &M->val, size_b*size_b*(M->numblocks) );
    magma_index_malloc( &M->row, r_blocks+1 );
    magma_index_malloc( &M->col, M->numblocks );

    // Prepare A 
    float **hA, **dA,  **hB, **dB, **hBL, **dBL, **hC, **dC;
    int rowt=0, rowt2=0;
    magma_malloc_cpu((void **)& hA, (M->numblocks)
                            *sizeof(float*) );
    magma_malloc(    (void **)&dA, (M->numblocks)
                            *sizeof(float*) );
    magma_malloc_cpu((void **)& hB, (M->numblocks)
                            *sizeof(float*) );
    magma_malloc(    (void **)&dB, (M->numblocks)
                            *sizeof(float*) );
    magma_malloc_cpu((void **)& hBL, (M->numblocks)
                            *sizeof(float*) );
    magma_malloc(    (void **)&dBL, (M->numblocks)
                            *sizeof(float*) );
    magma_malloc_cpu((void **)& hC, (M->numblocks)
                            *sizeof(float*) );
    magma_malloc(    (void **)&dC, (M->numblocks)
                            *sizeof(float*) );

    float **AIs, **AIIs, **dAIs, **dAIIs, **BIs, **dBIs;

    magma_malloc_cpu((void **)&AIs, r_blocks*c_blocks
                        *sizeof(float*));
    magma_malloc_cpu((void **)&BIs, r_blocks*c_blocks
                        *sizeof(float*));
    magma_malloc((void **)&dAIs, r_blocks*c_blocks
                        *sizeof(float*));
    magma_malloc((void **)&dBIs, r_blocks*c_blocks
                        *sizeof(float*));
    magma_malloc_cpu((void **)&AIIs, r_blocks*c_blocks
                        *sizeof(float*));
    magma_malloc((void **)&dAIIs, r_blocks*c_blocks
                        *sizeof(float*));

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
    magma_setvector( A.numblocks, sizeof(float*), 
                                                        hA, 1, dA, 1 );
    magma_setvector( A.numblocks, sizeof(float*), 
                                                        hB, 1, dB, 1 );
    magma_setvector( (M->numblocks-A.numblocks), sizeof(float*), hC, 1, dC, 1 );

    magma_sbcsrvalcpy(  size_b, A.numblocks, (M->numblocks-A.numblocks), 
                                                        dA, dB, dC );

    num_blocks_tmp=0;
    magma_index_t *cpu_row, *cpu_col;
    magma_index_malloc_cpu( &cpu_row, r_blocks+1 );
    magma_index_malloc_cpu( &cpu_col, M->numblocks );

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

    magma_index_setvector( r_blocks+1,   cpu_row, 1, M->row, 1 );            
    magma_index_setvector( M->numblocks, cpu_col, 1, M->col, 1 );
    magma_free_cpu( cpu_row );
    magma_free_cpu( cpu_col );

    magma_int_t ldda, lddb, lddc, ldwork;

    ldda = size_b;//((size_b+31)/32)*32;
    lddb = size_b;//((size_b+31)/32)*32;
    lddc = size_b;//((size_b+31)/32)*32;

    float *dwork;
    ldwork = size_b * magma_get_sgetri_nb( size_b );
    magma_smalloc( &dwork, ldwork );

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
        magma_setvector( klblocks, sizeof(float*), hBL, 1, 
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
        magma_setvector( num_block_rows, sizeof(float*), hA, 1, 
                                                        dA, 1 );
        magma_setvector( kblocks, sizeof(float*), hB, 1, dB, 1 );
            
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
        magma_setvector( kblocks*num_block_rows, sizeof(float*), 
                                                                hC, 1, dC, 1 );

        if( version==0 ){ 
        // AIs and BIs for the batched GEMMs later
            for(i=0; i<num_block_rows; i++){
               for(j=0; j<kblocks; j++){
                  AIs[j+i*kblocks] = hA[i];
                  BIs[j+i*kblocks] = hB[j];
                }
            }
            magma_setvector( kblocks*num_block_rows, sizeof(float*), AIs, 1, dAIs, 1 );
            magma_setvector( kblocks*num_block_rows, sizeof(float*), BIs, 1, dBIs, 1 );
        }  
        // AIIs for the batched TRSMs under the factorized block
        for(i=0; i<max(num_block_rows, kblocks); i++){
           AIIs[i] = M(k,k);
        }
        magma_setvector( max(num_block_rows, kblocks), 
                sizeof(float*), AIIs, 1, dAIIs, 1 );
        
        magma_sgetrf_gpu( size_b, size_b, M(k,k), ldda, ipiv+k*size_b, &info );


        // Swap elements on the right before update
        magma_isetvector( size_b, ipiv+k*size_b, 1, ipiv_d, 1 ); 
        magma_sbcsrlupivloc( size_b, kblocks, dB, ipiv_d );


        // update blocks right
        cublasStrsmBatched( handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER, 
                            CUBLAS_OP_N, CUBLAS_DIAG_UNIT, size_b, size_b, 
                            &one, dAIIs, size_b, dB, size_b, kblocks );



        // Swap elements on the left (anytime - hidden by the Ztrsm)
        magmablasSetKernelStream( stream[1] );
        magma_sbcsrlupivloc( size_b, klblocks, dBL, ipiv_d );



        // update blocks below
        cublasStrsmBatched( handle, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_UPPER, 
                            CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, size_b, size_b, 
                            &one, dAIIs, size_b, dA, size_b, num_block_rows );
        
        if( version==1 ){ 
            magmablasSetKernelStream( stream[0] );
            magma_sbcsrluegemm( size_b, num_block_rows, kblocks, dA, dB, dC ); 
        }

        if( version==0 ){ 
            //------------------------------------------------------------------
            // update trailing matrix using cublas batched GEMM
            magmablasSetKernelStream( stream[0] );
            cublasSgemmBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N, size_b, size_b, 
                              size_b,&m_one, (const float **) dAIs,  
                              size_b, (const float **) dBIs, 
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
}   /* magma_sbcsrlusv */


/**
    Purpose
    -------

    LU solve for a BCSR matrix A. 

    Arguments
    ---------

    @param
    A           magma_s_sparse_matrix
                input matrix A (on DEV)
                
    @param
    b           magma_s_vector
                RHS (on DEV)
                
    @param
    x           magma_s_vector
                solution (on DEV)

    @param
    solver_par  magma_s_solver_par*
                solver parameters

    @param
    ipiv        magma_int_t*
                pivot vector


    @ingroup magmasparse_sgesv
    ********************************************************************/

magma_int_t
magma_sbcsrlusv( magma_s_sparse_matrix A, magma_s_vector b, magma_s_vector *x,  
           magma_s_solver_par *solver_par, magma_int_t *ipiv ){

    // some useful variables
    magma_int_t size_b = A.blocksize;
    magma_int_t c_blocks = ceil( (float)A.num_cols / (float)size_b );     
                                        // max number of blocks per row
    magma_int_t r_blocks = ceil( (float)A.num_rows / (float)size_b );     
                                        // max number of blocks per column

    // set x = b
    magma_scopyvector( A.num_rows, b.val, 1, x->val, 1 );

    // First pivot the RHS
    magma_sbcsrswp( r_blocks, size_b, ipiv, x->val );


    // forward solve
    magma_sbcsrtrsv( MagmaLower, r_blocks, c_blocks, size_b, 
                     A.val, A.blockinfo, x->val );

    // backward solve
    magma_sbcsrtrsv( MagmaUpper, r_blocks, c_blocks, size_b, 
                     A.val, A.blockinfo, x->val );

    return MAGMA_SUCCESS;
}   /* magma_sbcsrlusv */



