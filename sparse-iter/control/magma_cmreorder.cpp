/*
    -- MAGMA (version 1.5.0-beta3) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2011

       @generated from magma_zmreorder.cpp normal z -> c, Fri Jul 18 17:34:30 2014
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

    Takes a matrix and a reordering scheme such that the output mat

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
    A           magma_c_sparse_matrix
                input/output matrix 

    @param
    n           magma_int_t
                nodes in one dimension

    @param
    b           magma_int_t
                blocksize

    @param
    B           magma_c_sparse_matrix*
                new matrix filled with new indices


    @ingroup magmasparse_c
    ********************************************************************/

extern "C" magma_int_t
magma_cmreorder( magma_c_sparse_matrix A, magma_int_t n, magma_int_t b, magma_c_sparse_matrix *B ){

if( A.memory_location == Magma_CPU && A.storage_type == Magma_CSRCOO ){
        
        magma_int_t entry, i, j, k, l;
        magma_c_mtransfer( A, B, Magma_CPU, Magma_CPU );

//magma_c_mvisu(A);


/*
        magma_index_t *p_h;
        magma_index_malloc_cpu( &p_h, A.num_rows );
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

                p_h[row] = count;
                //printf("row: %d+(%d+%d)*%d+%d=%d\n",(i+b1)*n*n,j,b2,n,(k+b3), row);
                for( entry=A.row[row]; entry<bound; entry++){
                    p_h[entry] = count;
                    count++;
                }
            }
            }
            }


        }// row
        }// i
        }// p
*/
        magma_index_t *p_h;
        magma_index_malloc_cpu( &p_h, A.nnz );

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

/*
        for(int i=0; i<A.nnz; i++)
            p_h[i] = i;
*/
        int limit=A.nnz;
/*

        for(int i=0; i< limit; i++){
            int idx1 = rand()%limit;
            int idx2 = rand()%limit;
            int tmp = p_h[idx1];
            p_h[idx1] = p_h[idx2];
            p_h[idx2] = tmp;
        }*/

        for( i=0; i<A.nnz; i++ ){
                B->val[p_h[i]] = A.val[i];
                B->col[p_h[i]] = A.col[i];
                B->rowidx[p_h[i]] = A.rowidx[i];
        }
/*
        count = 0;
        magma_int_t rowcount = 0;
        for( magma_int_t i=0; i<n; i+=b){
        for( magma_int_t j=0; j<n; j+=b){
        for( magma_int_t k=0; k<n; k+=b){

            for(magma_int_t b1=0; b1<b; b1++){
            for(magma_int_t b2=0; b2<b; b2++){
            for(magma_int_t b3=0; b3<b; b3++){
                magma_int_t row=(i+b1)*n*n+(j+b2)*n+(k+b3);
                magma_int_t bound = ( (row+1) < A.num_rows+1 ) ? 
                                            A.row[row+1] : A.nnz;
                
                //B->row[rowcount] = count;
                printf("row: %d-> %d\n",row, p_h[row] );
                for( entry=A.row[row]; entry<bound; entry++){
                    B->val[count] = A.val[entry];
                    B->col[count] = p_h[A.col[entry]];
                    B->rowidx[count] = p_h[A.rowidx[entry]];
                    count++;
                }
                rowcount++;
            }
            }
            }


        }// row
        }// i
        }// p
        //B->row[rowcount] = count;*/


   //for(i=0; i<100; i++)
    //printf("%d \n", p_h[i]);
        //magma_setvector( A.nnz, sizeof(magma_index_t), p_h, 1, p, 1 );
        magma_free_cpu( p_h );

        return MAGMA_SUCCESS; 
    }
    else{

        magma_c_sparse_matrix hA, CSRA;
        magma_c_mtransfer( A, &hA, A.memory_location, Magma_CPU );
        magma_c_mconvert( hA, &CSRA, hA.storage_type, Magma_CSRCOO );

        magma_cmreorder( CSRA, n, b, B );

        magma_c_mfree( &hA );
        magma_c_mfree( &CSRA );    

        return MAGMA_SUCCESS; 
    }
}





