/*
    -- MAGMA (version 1.5.0-beta3) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2011

       @generated from magma_zmhom.cpp normal z -> s, Fri Jul 18 17:34:30 2014
       @author Hartwig Anzt

*/
#include "magma_lapack.h"
#include "common_magma.h"
#include "../include/magmasparse.h"

#include <assert.h>

// includes CUDA
#include <cuda_runtime_api.h>
#include <cublas.h>
#include <cusparse_v2.h>
#include <cuda_profiler_api.h>

#define RTOLERANCE     lapackf77_slamch( "E" )
#define ATOLERANCE     lapackf77_slamch( "E" )

#define min(a, b) ((a) < (b) ? (a) : (b))

/** -- MAGMA (version 1.5.0-beta3) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2011

    Purpose
    -------

    Takes a matrix and a blocksize b to generate a homomorphism that
    orders the matrix entries according to the subdomains of size b x b.
    Returns p on the device

    example:

        / a 0 0 b 0 \
        | 0 c 0 d 0 |
     A= | 0 e f g 0 |       b = 2
        | h 0 0 0 0 |
        \ i j 0 0 0 /

    will generate the projection:
    
    0 2 1 3 4 7 8 9 10 11
    
    according to
    
    a c b d e h f g i j    

    Arguments
    ---------

    @param
    A           magma_s_sparse_matrix
                input/output matrix 

    @param
    b           magma_int_t
                blocksize

    @param
    p           magma_index_t*
                homomorphism vector containing the indices


    @ingroup magmasparse_saux
    ********************************************************************/

extern "C" magma_int_t
magma_smhom( magma_s_sparse_matrix A, magma_int_t b, magma_index_t *p ){

    if( A.memory_location == Magma_CPU && A.storage_type == Magma_CSRCOO ){
        
        magma_int_t i, j, k, l;
        magma_int_t rblock, r_blocks = (A.num_rows+b-1)/b;
        magma_int_t cblock, c_blocks = (A.num_cols+b-1)/b; 

        magma_index_t *p_h;
        magma_index_malloc_cpu( &p_h, A.nnz );

//magma_s_mvisu(A);

        j=0;
        for( rblock=0; rblock<r_blocks; rblock++){
        for( cblock=0; cblock<c_blocks; cblock++){
            magma_int_t bound = A.nnz;
            bound = ( (rblock+1)*b < A.num_rows ) ? 
                                            A.row[(rblock+1)*b] : A.nnz;
            for( i=A.row[rblock*b]; i<bound; i++){
                if( ( cblock*b <= A.col[i] && A.col[i] < (cblock+1)*b )
                        && 
                   ( rblock*b <= A.rowidx[i] && A.rowidx[i] < (rblock+1)*b ) ){

//printf("insert %f -> %d because rblock:%d cblock%d b:%d A.col:%d A.rowidx:%d\n", A.val[i], i, rblock, cblock, b, A.col[i], A.rowidx[i]);

                    // insert this index at this point at the homomorphism
                        p_h[j] = i;
                        j++;
                 }

            }
        }// cblocks
        }// rblocks

        magma_setvector( A.nnz, sizeof(magma_index_t), p_h, 1, p, 1 );
        magma_free_cpu( p_h );
        return MAGMA_SUCCESS; 
    }
    else{

        magma_s_sparse_matrix hA, CSRA;
        magma_s_mtransfer( A, &hA, A.memory_location, Magma_CPU );
        magma_s_mconvert( hA, &CSRA, hA.storage_type, Magma_CSRCOO );

        magma_smhom( CSRA, b, p );

        magma_s_mfree( &hA );
        magma_s_mfree( &CSRA );    

        return MAGMA_SUCCESS; 
    }
}







/** -- MAGMA (version 1.5.0-beta3) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2011

    Purpose
    -------

    Takes a matrix and a blocksize b to generate a homomorphism that
    orders the matrix entries according to the subdomains of size b x b.
    Returns p on the device

    example:

        / a 0 0 b 0 \
        | 0 c 0 d 0 |
     A= | 0 e f g 0 |       b = 2
        | h 0 0 0 0 |
        \ i j 0 0 0 /

    will generate the projection:
    
    0 2 1 3 4 7 8 9 10 11
    
    according to
    
    a c b d e h f g i j    

    Arguments
    ---------

    @param
    A           magma_s_sparse_matrix
                input/output matrix 

    @param
    b           magma_int_t
                blocksize

    @param
    p           magma_index_t*
                homomorphism vector containing the indices


    @ingroup magmasparse_s
    ********************************************************************/

extern "C" magma_int_t
magma_smhom_fd( magma_s_sparse_matrix A, magma_int_t n, magma_int_t b, magma_index_t *p ){

    if( A.memory_location == Magma_CPU ){
        
        magma_int_t entry, i, j, k, l;
        magma_int_t rblock, r_blocks = (A.num_rows+b-1)/b;
        magma_int_t cblock, c_blocks = (A.num_cols+b-1)/b; 

        magma_index_t *p_h;
        magma_index_malloc_cpu( &p_h, A.nnz );

//magma_s_mvisu(A);


        magma_int_t count=0;
        for( magma_int_t i=0; i<n; i+=b){
        for( magma_int_t j=0; j<n; j+=b){
        for( magma_int_t k=0; k<n; k+=b){

            for(magma_int_t b1=0; b1<b; b1++){
            for(magma_int_t b2=0; b2<b; b2++){
            for(magma_int_t b3=0; b3<b; b3++){
                magma_int_t row=(i+b1)*n*n+(j+b2)*n+(k+b3);
                magma_int_t bound = ( (row+1) < A.num_rows+1 ) ? 
                                            A.row[row+1] : A.nnz;
                //printf("row: %d+(%d+%d)*%d+%d=%d\n",(i+b1)*n*n,j,b2,n,(k+b3), row);
                for( entry=A.row[row]; entry<bound; entry++){
                    p_h[count] = entry;
                    count++;
                }
            }
            }
            }


        }// row
        }// i
        }// p

  //  for(i=0; i<A.nnz; i++)
  //  printf("%d \n", p_h[i]);
        magma_setvector( A.nnz, sizeof(magma_index_t), p_h, 1, p, 1 );
        magma_free_cpu( p_h );
        return MAGMA_SUCCESS; 
    }
    else{

        magma_s_sparse_matrix hA, CSRA;
        magma_s_mtransfer( A, &hA, A.memory_location, Magma_CPU );
        magma_s_mconvert( hA, &CSRA, hA.storage_type, Magma_CSRCOO );

        magma_smhom( CSRA, b, p );

        magma_s_mfree( &hA );
        magma_s_mfree( &CSRA );    

        return MAGMA_SUCCESS; 
    }
}



/** -- MAGMA (version 1.5.0-beta3) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2011

    Purpose
    -------

    Takes a matrix and a blocksize b to generate a homomorphism that
    orders the matrix entries according to the subdomains of size b x b.
    Returns p on the device

    example:

        / a 0 0 b 0 \
        | 0 c 0 d 0 |
     A= | 0 e f g 0 |       b = 2
        | h 0 0 0 0 |
        \ i j 0 0 0 /

    will generate the projection:
    
    0 2 1 3 4 7 8 9 10 11
    
    according to
    
    a c b d e h f g i j    

    Arguments
    ---------

    @param
    A           magma_s_sparse_matrix
                input/output matrix 

    @param
    b           magma_int_t
                blocksize

    @param
    p           magma_index_t*
                homomorphism vector containing the indices


    @ingroup magmasparse_s
    ********************************************************************/
/*
extern "C" magma_int_t
magma_smhom_fd( magma_s_sparse_matrix A, magma_int_t n, magma_int_t b, magma_index_t *p ){

    if( A.memory_location == Magma_CPU ){
        
        magma_int_t entry, i, j, k, l;
        magma_int_t rblock, r_blocks = (A.num_rows+b-1)/b;
        magma_int_t cblock, c_blocks = (A.num_cols+b-1)/b; 

        magma_index_t *p_h;
        magma_index_malloc_cpu( &p_h, A.nnz );

//magma_s_mvisu(A);


        j=0;
        for( magma_int_t p=0; p<n/b; p++){
        for( magma_int_t i=0; i<A.num_rows/n; i++){
        for( magma_int_t row=i*n+p*b; row<i*n+p*b+b; row++ ){

        magma_int_t bound = ( (row+1) < A.num_rows+1 ) ? 
                                            A.row[row+1] : A.nnz;

        for( entry=A.row[row]; entry<bound; entry++){
            //printf("insert %d at %d row:%d i:%d\n", entry, j, row, i);
            p_h[j] = entry;
            j++;
        }// entry
              // printf("row: %d\n", row);
        }// row
        }// i
        }// p

  //  for(i=0; i<A.nnz; i++)
  //  printf("%d \n", p_h[i]);
        magma_setvector( A.nnz, sizeof(magma_index_t), p_h, 1, p, 1 );
        magma_free_cpu( p_h );
        return MAGMA_SUCCESS; 
    }
    else{

        magma_s_sparse_matrix hA, CSRA;
        magma_s_mtransfer( A, &hA, A.memory_location, Magma_CPU );
        magma_s_mconvert( hA, &CSRA, hA.storage_type, Magma_CSRCOO );

        magma_smhom( CSRA, b, p );

        magma_s_mfree( &hA );
        magma_s_mfree( &CSRA );    

        return MAGMA_SUCCESS; 
    }
}
*/



